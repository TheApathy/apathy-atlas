// SPDX-License-Identifier: AGPL-3.0-only

//! Incremental suffix automaton over a sequence's own token history.
//!
//! This is the self-context drafting tier: the corpus is the sequence
//! itself — prompt plus everything generated so far. No file, no network,
//! and it works on prompts nothing was ever indexed for, because the
//! thing being matched against is the generation in progress.
//!
//! # Why an automaton
//!
//! The naive version (`spark-server::ngram`) rescans the whole history for
//! the longest suffix match on every decode step: O(n·k) comparisons per
//! step, on the decode thread, against a history that reaches tens of
//! thousands of tokens. A suffix automaton is the textbook structure for
//! the same query at O(1) amortized per appended token, and it carries a
//! second gift: the *match state* can be maintained incrementally too, so
//! the longest-suffix-match at the current position is already known
//! before the query is asked. See Blumer et al.; the speculative-decoding
//! application mirrors SAM-Decoding (2025).
//!
//! # What it answers
//!
//! After appending tokens `t[0..n)`, [`SuffixAutomaton::match_len`] and
//! [`SuffixAutomaton::match_end`] describe the longest suffix of `t[0..n)`
//! that also occurs **earlier** in `t`, and where that earlier occurrence
//! ended. The continuation after that earlier occurrence is the draft.
//! [`SuffixAutomaton::peek`] asks the same question for one hypothetical
//! next token without committing it, which is what the scheduler needs:
//! it has just sampled a token that is not in the committed history yet.
//!
//! # Ordering invariant
//!
//! [`SuffixAutomaton::push`] advances the match state *before* extending
//! the automaton, so a match can never resolve to the token just added.
//! Reversing those two lines would let every position match itself, report
//! a full-length match, and propose the tokens that follow the current
//! position — which do not exist yet. The tests pin the ordering.

/// A state's outgoing transitions.
///
/// Small states dominate (most have one or two transitions), so a sorted
/// vector beats a hash map on both memory and lookup until a state gets
/// wide; the root is the only reliably wide one. `Vec` of pairs keeps the
/// per-state overhead at 24 bytes plus 8 per edge.
type Transitions = Vec<(u32, u32)>;

/// One automaton state.
struct State {
    /// Length of the longest string in this state's equivalence class.
    len: u32,
    /// Suffix link, or `u32::MAX` for the root.
    link: u32,
    /// End position (exclusive) of the most recent occurrence of this
    /// state's longest string. Cloned states inherit it, which is what
    /// makes the continuation lookup correct for shorter matches too.
    end: u32,
    /// Sorted by token id.
    next: Transitions,
}

impl State {
    fn get(&self, token: u32) -> Option<u32> {
        self.next
            .binary_search_by_key(&token, |&(t, _)| t)
            .ok()
            .map(|i| self.next[i].1)
    }

    fn set(&mut self, token: u32, to: u32) {
        match self.next.binary_search_by_key(&token, |&(t, _)| t) {
            Ok(i) => self.next[i].1 = to,
            Err(i) => self.next.insert(i, (token, to)),
        }
    }
}

/// Incremental suffix automaton plus the running longest-suffix match.
pub struct SuffixAutomaton {
    states: Vec<State>,
    /// The state representing the whole string so far.
    last: u32,
    /// Tokens appended, i.e. the length of the indexed history.
    len: u32,
    /// Current match state and its length — the longest suffix of the
    /// history that occurred earlier.
    match_state: u32,
    match_len: u32,
}

impl Default for SuffixAutomaton {
    fn default() -> Self {
        Self::new()
    }
}

impl SuffixAutomaton {
    /// An automaton over the empty history.
    pub fn new() -> Self {
        let root = State {
            len: 0,
            link: u32::MAX,
            end: 0,
            next: Vec::new(),
        };
        Self {
            states: vec![root],
            last: 0,
            len: 0,
            match_state: 0,
            match_len: 0,
        }
    }

    /// Tokens indexed so far.
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Whether nothing has been indexed.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Length of the longest suffix of the history that occurs earlier.
    pub fn match_len(&self) -> usize {
        self.match_len as usize
    }

    /// End position (exclusive) of that earlier occurrence, or `None` when
    /// there is no match.
    pub fn match_end(&self) -> Option<usize> {
        (self.match_len > 0).then(|| self.states[self.match_state as usize].end as usize)
    }

    /// Approximate heap footprint in bytes. Reported so the per-sequence
    /// memory cost can be stated rather than guessed.
    pub fn heap_bytes(&self) -> usize {
        let per_state = std::mem::size_of::<State>();
        let edges: usize = self.states.iter().map(|s| s.next.capacity()).sum();
        self.states.capacity() * per_state + edges * std::mem::size_of::<(u32, u32)>()
    }

    /// Walk the match state over `token` without mutating anything.
    ///
    /// Returns the `(match_len, end)` that [`push`](Self::push) would
    /// produce for this token: the longest suffix of `history ++ [token]`
    /// occurring in the history, and where that occurrence ended.
    /// `match_len == 0` means no occurrence.
    pub fn peek(&self, token: u32) -> (usize, usize) {
        let (state, len) = self.advance(self.match_state, self.match_len, token);
        if len == 0 {
            return (0, 0);
        }
        (len as usize, self.states[state as usize].end as usize)
    }

    /// Append one token: advance the match, then extend the automaton.
    pub fn push(&mut self, token: u32) {
        // Order matters — see the module docs. The match is resolved
        // against the automaton BEFORE `token` enters it, so it can only
        // point at an earlier occurrence.
        let (state, len) = self.advance(self.match_state, self.match_len, token);
        self.match_state = state;
        self.match_len = len;
        self.extend(token);
    }

    /// Append many tokens. Accepted speculative batches arrive several at
    /// a time; this is just the loop, kept here so callers cannot get the
    /// per-token ordering wrong.
    pub fn extend_from_slice(&mut self, tokens: &[u32]) {
        for &t in tokens {
            self.push(t);
        }
    }

    /// Follow transitions and suffix links to extend a match by `token`.
    fn advance(&self, mut state: u32, mut len: u32, token: u32) -> (u32, u32) {
        loop {
            if let Some(next) = self.states[state as usize].get(token) {
                return (next, len + 1);
            }
            let link = self.states[state as usize].link;
            if link == u32::MAX {
                // Fell back to the root with no transition: no occurrence.
                return (0, 0);
            }
            state = link;
            len = self.states[state as usize].len;
        }
    }

    fn new_state(&mut self, len: u32, link: u32, end: u32, next: Transitions) -> u32 {
        self.states.push(State {
            len,
            link,
            end,
            next,
        });
        (self.states.len() - 1) as u32
    }

    /// Standard online SAM construction (Blumer), with `end` maintained.
    fn extend(&mut self, token: u32) {
        self.len += 1;
        let cur = self.new_state(
            self.states[self.last as usize].len + 1,
            u32::MAX,
            self.len,
            Vec::new(),
        );

        let mut p = self.last;
        loop {
            if self.states[p as usize].get(token).is_some() {
                break;
            }
            self.states[p as usize].set(token, cur);
            match self.states[p as usize].link {
                u32::MAX => {
                    self.states[cur as usize].link = 0;
                    self.last = cur;
                    return;
                }
                link => p = link,
            }
        }

        let q = self.states[p as usize]
            .get(token)
            .expect("loop exits only on an existing transition");
        if self.states[p as usize].len + 1 == self.states[q as usize].len {
            self.states[cur as usize].link = q;
            self.last = cur;
            return;
        }

        // Split `q` into a clone carrying the shorter class.
        let clone = self.new_state(
            self.states[p as usize].len + 1,
            self.states[q as usize].link,
            self.states[q as usize].end,
            self.states[q as usize].next.clone(),
        );
        while self.states[p as usize].get(token) == Some(q) {
            self.states[p as usize].set(token, clone);
            match self.states[p as usize].link {
                u32::MAX => break,
                link => p = link,
            }
        }
        self.states[q as usize].link = clone;
        self.states[cur as usize].link = clone;
        self.last = cur;

        // A split can shorten the current match's class; keep the match
        // state pointing at the class that still contains it.
        if self.match_state == q && self.match_len <= self.states[clone as usize].len {
            self.match_state = clone;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementation: the longest suffix of `t[..n]` that also
    /// occurs ending at some earlier position, and that occurrence's end.
    fn brute(t: &[u32]) -> (usize, usize) {
        let n = t.len();
        for len in (1..n).rev() {
            let suffix = &t[n - len..];
            for end in len..n {
                if &t[end - len..end] == suffix {
                    return (len, end);
                }
            }
        }
        (0, 0)
    }

    fn build(tokens: &[u32]) -> SuffixAutomaton {
        let mut sam = SuffixAutomaton::new();
        sam.extend_from_slice(tokens);
        sam
    }

    #[test]
    fn empty_history_matches_nothing() {
        let sam = SuffixAutomaton::new();
        assert_eq!(sam.match_len(), 0);
        assert_eq!(sam.match_end(), None);
        assert_eq!(sam.peek(7), (0, 0));
    }

    #[test]
    fn a_repeated_span_matches_its_earlier_occurrence() {
        // [1,2,3,4, 1,2,3] — the suffix [1,2,3] occurred ending at 3.
        let sam = build(&[1, 2, 3, 4, 1, 2, 3]);
        assert_eq!(sam.match_len(), 3);
        assert_eq!(sam.match_end(), Some(3));
        // The continuation there is token 4, which is the draft.
        assert_eq!(
            sam.peek(4),
            (4, 4),
            "extending the match by its true successor"
        );
    }

    #[test]
    fn a_match_never_resolves_to_the_current_position() {
        // Every token distinct: nothing can match except by self-match,
        // which the push ordering forbids.
        let sam = build(&[10, 11, 12, 13, 14, 15]);
        assert_eq!(sam.match_len(), 0);
        assert_eq!(sam.match_end(), None);
    }

    #[test]
    fn peek_does_not_mutate() {
        let mut sam = build(&[5, 6, 7, 5, 6]);
        let before = (sam.match_len(), sam.match_end(), sam.len());
        let peeked = sam.peek(7);
        assert_eq!((sam.match_len(), sam.match_end(), sam.len()), before);
        // Committing the peeked token reproduces exactly what peek said.
        sam.push(7);
        assert_eq!((sam.match_len(), sam.match_end().unwrap()), peeked);
    }

    #[test]
    fn matches_the_brute_force_reference_on_every_prefix() {
        // Small alphabet so matches are dense, plus a couple of long
        // repeats to exercise the clone path.
        let mut tokens: Vec<u32> = Vec::new();
        let mut x: u32 = 7;
        for _ in 0..400 {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            tokens.push((x >> 16) % 5);
        }
        tokens.extend_from_slice(&[1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0]);

        let mut sam = SuffixAutomaton::new();
        for n in 0..tokens.len() {
            sam.push(tokens[n]);
            let (want_len, want_end) = brute(&tokens[..n + 1]);
            assert_eq!(
                sam.match_len(),
                want_len,
                "match length at prefix {}",
                n + 1
            );
            if want_len > 0 {
                // The reference returns the earliest occurrence; the
                // automaton returns the most recent one. Both are valid
                // occurrences — assert the SPAN matches, not the index.
                let end = sam.match_end().expect("a match has an end");
                assert_eq!(
                    &tokens[end - want_len..end],
                    &tokens[n + 1 - want_len..n + 1],
                    "the reported occurrence must really be that suffix (prefix {})",
                    n + 1
                );
                assert!(end <= n, "occurrence must end before the current position");
                let _ = want_end;
            }
        }
    }

    #[test]
    fn peek_agrees_with_brute_force_on_a_hypothetical_token() {
        let tokens: Vec<u32> = vec![1, 2, 3, 1, 2, 3, 4, 1, 2];
        let sam = build(&tokens);
        for probe in 0..6u32 {
            let (len, end) = sam.peek(probe);
            let mut extended = tokens.clone();
            extended.push(probe);
            let (want, _) = brute(&extended);
            assert_eq!(len, want, "peek({probe})");
            if len > 0 {
                assert!(end <= tokens.len(), "the occurrence must predate the probe");
                assert_eq!(
                    &tokens[end - len..end],
                    &extended[extended.len() - len..],
                    "peek({probe}) must report a real earlier occurrence"
                );
            }
        }
    }

    #[test]
    fn heap_growth_is_linear_in_history() {
        let a = build(&(0..1000u32).map(|i| i % 97).collect::<Vec<_>>());
        let b = build(&(0..2000u32).map(|i| i % 97).collect::<Vec<_>>());
        assert!(
            b.heap_bytes() < 3 * a.heap_bytes(),
            "doubling the history must not more than triple the footprint: {} vs {}",
            a.heap_bytes(),
            b.heap_bytes()
        );
    }
}
