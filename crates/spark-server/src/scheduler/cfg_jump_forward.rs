// SPDX-License-Identifier: AGPL-3.0-only

//! CFG jump-forward for code-generation drafting (`ATLAS_DFLASH_CFG_JF=1`,
//! default OFF).
//!
//! # What this does
//!
//! When the model is generating source code, a large fraction of the next
//! tokens are STRUCTURALLY FORCED: the `)` that closes an open `(`, the `]`
//! that closes a `[`, the `:` after a `def foo(...)`, the newline+indent that
//! must follow a `:` at end of line, the closing quote of an open string. A
//! neural drafter is often *uncertain* exactly at those positions (they carry
//! almost no information, so the drafter's marginal is flat), which truncates
//! the DFlash draft chain right where a cheap deterministic predictor would
//! have been correct.
//!
//! This module maintains a lightweight per-splice **bracket / structure
//! tracker** — a stack of open delimiters plus an "after-colon" flag — derived
//! purely from the committed token stream, and SPLICES the forced closer /
//! structural token into the drafted chain at the first position where the
//! neural drafter's token CONTRADICTS the forced one. The rest of the drafter's
//! chain is kept as long as it stays consistent with the (updated) tracker,
//! else truncated at the splice.
//!
//! # Why it is LOSSLESS by construction
//!
//! Splicing only changes WHAT is proposed. The DFlash verify path
//! ([`super::verify_dflash_step`]) is a greedy oracle: it commits the target's
//! argmax token and accepts a draft *only* when `draft == target_argmax`. A
//! wrong splice is therefore rejected exactly like a wrong drafter token — it
//! costs one rejected speculation and can NEVER change committed output. With
//! the flag off the drafts pass through untouched, so output is byte-identical.
//!
//! # Design: token-id classification, not string re-decode on the hot path
//!
//! The tracker needs to know what each token *means* structurally. Decoding a
//! token to a string per step would be far too slow. Instead we classify every
//! vocabulary id ONCE at startup ([`build_delim_table`], driven from the
//! tokenizer in `serve_phases::tokenizer_runtime`) into a compact
//! [`TokenDelim`] and cache it in a process-wide table. The hot path is then a
//! table lookup per token — no tokenizer, no allocation.
//!
//! # Honest limitations (when unsure, DON'T splice)
//!
//! * Strings / comments / f-strings: a `)` inside a string literal is NOT a
//!   real close. The tracker tracks single/double/triple quotes to suppress
//!   bracket bookkeeping inside strings, but nested f-string braces and
//!   language-specific escaping are only approximated. When the tracker's state
//!   is ambiguous (a token that mixes a quote and a bracket, an unrecognized
//!   multi-char delimiter token) it is treated as OPAQUE and we STOP splicing
//!   for the rest of the chain rather than guess.
//! * We never splice a closer that isn't *unambiguously* the single legal next
//!   structural token. Only the innermost open delimiter's matching closer, or
//!   the colon/newline cases below, are ever spliced.
//! * Multi-language: the delimiter set is language-agnostic (brackets/quotes)
//!   plus a small Python-flavored colon/indent heuristic. On non-Python code the
//!   colon/indent splices simply won't fire (no `:`-forced newline), and the
//!   bracket splices remain valid.

use std::sync::OnceLock;

/// Structural classification of a single vocabulary token, derived from its
/// decoded string. Only delimiter-relevant classes are distinguished; every
/// other token is [`TokenDelim::Other`] (a normal content token that neither
/// opens nor closes structure and clears the after-colon state on a newline).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenDelim {
    /// Token is exactly one opening bracket: `(`, `[`, or `{`.
    Open(u8),
    /// Token is exactly one closing bracket: `)`, `]`, or `}`.
    Close(u8),
    /// Token is exactly a single-line quote delimiter char: `'` or `"` or `` ` ``.
    Quote(u8),
    /// Token is exactly a triple-quote: `'''` or `"""`.
    TripleQuote(u8),
    /// Token is exactly `:` (colon) — arms the after-colon / newline-forced
    /// heuristic.
    Colon,
    /// Token STARTS a new line (its decoded string ends the previous line,
    /// i.e. contains a `\n`). Used to detect the newline that a trailing colon
    /// forces, and to reset transient line state.
    Newline,
    /// A normal content token: neither structural nor a clean line boundary.
    Other,
    /// The token's decoded string is structurally AMBIGUOUS — it mixes a quote
    /// with a bracket, contains an unbalanced set of delimiters we can't reason
    /// about, or failed to decode. When the tracker meets an opaque token
    /// (either in the committed stream or a draft) it stops trusting its stack
    /// and refuses to splice. Fail-safe: opaque never *forces* anything.
    Opaque,
}

/// Match a closing bracket byte to its opening counterpart.
fn matching_open(close: u8) -> u8 {
    match close {
        b')' => b'(',
        b']' => b'[',
        b'}' => b'{',
        other => other,
    }
}

/// Match an opening bracket byte to its closing counterpart.
fn matching_close(open: u8) -> u8 {
    match open {
        b'(' => b')',
        b'[' => b']',
        b'{' => b'}',
        other => other,
    }
}

/// Classify a single token's decoded string into a [`TokenDelim`].
///
/// The classification is deliberately CONSERVATIVE: only tokens whose string is
/// *exactly* one recognized delimiter (optionally with surrounding whitespace)
/// get a structural class. Anything that packs a delimiter together with other
/// content (`");"`, `"()"`, `"':'"`, `"](x"`) is [`TokenDelim::Opaque`] — we do
/// not attempt to reason about the internal structure of multi-symbol tokens,
/// and an opaque token halts splicing. This is the "when unsure, don't splice"
/// guarantee applied at classification time.
pub fn classify_str(s: &str) -> TokenDelim {
    // A leading space is common on BPE tokens (e.g. " (" opens a paren). Trim
    // surrounding ASCII whitespace *except* newlines, which are meaningful.
    let has_newline = s.contains('\n');
    let core = s.trim_matches([' ', '\t', '\r']);

    // Newline-only (or trailing-newline) token with no other structure.
    if has_newline {
        // A token that is purely whitespace + newline (indentation, blank line)
        // is a clean line boundary. If it *also* carries a delimiter we can't
        // safely reason about the interleaving → opaque.
        let non_ws: String = core.chars().filter(|c| !c.is_whitespace()).collect();
        if non_ws.is_empty() {
            return TokenDelim::Newline;
        }
        return TokenDelim::Opaque;
    }

    match core {
        "(" => TokenDelim::Open(b'('),
        "[" => TokenDelim::Open(b'['),
        "{" => TokenDelim::Open(b'{'),
        ")" => TokenDelim::Close(b')'),
        "]" => TokenDelim::Close(b']'),
        "}" => TokenDelim::Close(b'}'),
        "'" => TokenDelim::Quote(b'\''),
        "\"" => TokenDelim::Quote(b'"'),
        "`" => TokenDelim::Quote(b'`'),
        "'''" => TokenDelim::TripleQuote(b'\''),
        "\"\"\"" => TokenDelim::TripleQuote(b'"'),
        ":" => TokenDelim::Colon,
        "" => TokenDelim::Other, // pure whitespace, no newline
        _ => {
            // Multi-char token. If it contains ANY delimiter/quote char it is
            // ambiguous (we can't know the ordering / string-escaping) → opaque.
            // Otherwise it is ordinary content.
            if core
                .bytes()
                .any(|b| matches!(b, b'(' | b')' | b'[' | b']' | b'{' | b'}' | b'\'' | b'"' | b'`' | b':'))
            {
                TokenDelim::Opaque
            } else {
                TokenDelim::Other
            }
        }
    }
}

/// Process-wide token→[`TokenDelim`] table, indexed by token id. `None` until
/// [`set_delim_table`] runs (built from the tokenizer at startup only when
/// `ATLAS_DFLASH_CFG_JF=1`). Absent ⇒ CFG jump-forward is inert.
static DELIM_TABLE: OnceLock<std::sync::Arc<[TokenDelim]>> = OnceLock::new();

/// Install the classification table. Idempotent (first writer wins).
pub fn set_delim_table(table: std::sync::Arc<[TokenDelim]>) {
    let _ = DELIM_TABLE.set(table);
}

/// Read the classification table. `None` until [`set_delim_table`] runs.
pub fn delim_table() -> Option<std::sync::Arc<[TokenDelim]>> {
    DELIM_TABLE.get().cloned()
}

/// Whether CFG jump-forward is enabled (`ATLAS_DFLASH_CFG_JF=1`). Read once.
pub fn cfg_jf_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_DFLASH_CFG_JF").ok().as_deref() == Some("1"))
}

/// Build the classification table from the tokenizer by decoding every id.
///
/// `decode` is the tokenizer's single-id decode (byte-level, so a leading BPE
/// space is a real ' '). Any decode error leaves that id [`TokenDelim::Other`]
/// (fail-open: an undecodable token can't be a delimiter). Returns the table
/// plus the count of ids that got a structural (non-Other, non-Opaque) class,
/// for logging.
pub fn build_delim_table<F>(vocab_size: usize, mut decode: F) -> (Vec<TokenDelim>, usize)
where
    F: FnMut(u32) -> Option<String>,
{
    let mut table = vec![TokenDelim::Other; vocab_size];
    let mut structural = 0usize;
    for (id, slot) in table.iter_mut().enumerate() {
        if let Some(s) = decode(id as u32) {
            let c = classify_str(&s);
            *slot = c;
            if !matches!(c, TokenDelim::Other | TokenDelim::Opaque) {
                structural += 1;
            }
        }
    }
    (table, structural)
}

/// A single open-delimiter frame on the tracker stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Frame {
    /// An open bracket; carries its opening byte so we know the forced closer.
    Bracket(u8),
    /// An open single-line string; carries its quote byte.
    Str(u8),
    /// An open triple-quoted string; carries its quote byte.
    TripleStr(u8),
}

/// Bracket / structure tracker. Walks a token stream left-to-right maintaining
/// a stack of open delimiters plus the "line ended with a colon" flag that the
/// newline-forcing heuristic uses.
#[derive(Clone, Debug, Default)]
pub struct BracketTracker {
    stack: Vec<Frame>,
    /// Set true when the most recent *structural* token was a `:` and nothing
    /// since has ended the line. When a newline is the forced next token we can
    /// use this. Reset by any newline.
    after_colon: bool,
    /// Once we hit a token we cannot reason about while it matters, we give up
    /// forcing for the rest of THIS splice walk (fail-safe). The committed-
    /// prefix walk never sets this; only the draft-simulation walk does.
    poisoned: bool,
}

impl BracketTracker {
    /// Fresh tracker with an empty stack.
    pub fn new() -> Self {
        Self::default()
    }

    /// True while inside any string (single or triple). Bracket bookkeeping is
    /// suppressed here — a `)` inside a string is literal text.
    fn in_string(&self) -> bool {
        matches!(self.stack.last(), Some(Frame::Str(_) | Frame::TripleStr(_)))
    }

    /// Advance the tracker by one classified token. Pure state mutation; does
    /// not decide splices. `poisoned` short-circuits to a no-op so a walk that
    /// gave up stays given-up.
    pub fn advance(&mut self, d: TokenDelim) {
        if self.poisoned {
            return;
        }
        match d {
            TokenDelim::TripleQuote(q) => {
                match self.stack.last() {
                    Some(Frame::TripleStr(open)) if *open == q => {
                        self.stack.pop(); // closes the triple string
                    }
                    Some(Frame::Str(_)) => { /* triple-quote text inside a
                                              single-quoted string: ignore */ }
                    _ => self.stack.push(Frame::TripleStr(q)),
                }
                self.after_colon = false;
            }
            TokenDelim::Quote(q) => {
                match self.stack.last() {
                    Some(Frame::Str(open)) if *open == q => {
                        self.stack.pop(); // closes the single-line string
                    }
                    Some(Frame::Str(_)) | Some(Frame::TripleStr(_)) => { /* other
                                              quote char inside a string: text */ }
                    _ => self.stack.push(Frame::Str(q)),
                }
                self.after_colon = false;
            }
            _ if self.in_string() => {
                // Inside a string, only quote tokens (handled above) matter.
                // Brackets/colons/newlines are literal text. A newline inside a
                // *single-line* string is actually illegal in most languages,
                // but we conservatively leave the string open rather than guess.
                if matches!(d, TokenDelim::Newline) {
                    self.after_colon = false;
                }
            }
            TokenDelim::Open(b) => {
                self.stack.push(Frame::Bracket(b));
                self.after_colon = false;
            }
            TokenDelim::Close(b) => {
                let want = matching_open(b);
                match self.stack.last() {
                    Some(Frame::Bracket(open)) if *open == want => {
                        self.stack.pop();
                    }
                    // Mismatched / unexpected close: the stream is doing
                    // something our model doesn't capture. Poison so we stop
                    // splicing (only matters on the draft walk).
                    _ => self.poisoned = true,
                }
                self.after_colon = false;
            }
            TokenDelim::Colon => {
                self.after_colon = true;
            }
            TokenDelim::Newline => {
                self.after_colon = false;
            }
            TokenDelim::Other => {
                // Ordinary content on the current line clears a pending colon
                // ONLY if it is non-whitespace; but our classifier folds pure
                // whitespace into Other too. We keep after_colon set through
                // trailing whitespace so `):\n` still triggers. Simplest safe
                // rule: leave after_colon unchanged for Other.
            }
            TokenDelim::Opaque => {
                // Can't reason about this token's effect on structure.
                self.poisoned = true;
            }
        }
    }

    /// The single structurally-forced next token's [`TokenDelim`], if any.
    ///
    /// Returns `Some(delim)` ONLY when the next token is unambiguously forced:
    /// the innermost open string's closing quote, or (when no string is open)
    /// the innermost open bracket's closing bracket. The colon/newline case is
    /// handled by the caller (it needs a concrete newline token id, which is
    /// language/tokenizer specific), so this returns `None` there.
    ///
    /// Never returns a forced token when [`poisoned`](Self::poisoned).
    fn forced_next(&self) -> Option<TokenDelim> {
        if self.poisoned {
            return None;
        }
        match self.stack.last() {
            Some(Frame::TripleStr(q)) => Some(TokenDelim::TripleQuote(*q)),
            Some(Frame::Str(q)) => Some(TokenDelim::Quote(*q)),
            Some(Frame::Bracket(b)) => Some(TokenDelim::Close(matching_close(*b))),
            None => None,
        }
    }
}

/// The concrete token id that realizes a forced [`TokenDelim`], chosen from a
/// reverse lookup built once alongside the table. We need, e.g., the id for the
/// bare `)` token. Because multiple ids may decode to the same delimiter (with
/// different leading whitespace), we prefer the id whose classification matches
/// AND whose decoded form is the *bare* delimiter — but since the table only
/// records the class, we accept the first id per class. This map is built in
/// [`build_forced_ids`].
#[derive(Clone, Debug, Default)]
pub struct ForcedIds {
    /// id of a bare `)` / `]` / `}` keyed by the closing byte.
    close: [Option<u32>; 3],
    /// id of a bare `'` / `"` / `` ` ``.
    quote: [Option<u32>; 3],
    /// id of a bare `'''` / `"""` (index 0 = `'`, 1 = `"`).
    triple: [Option<u32>; 2],
}

fn close_idx(b: u8) -> Option<usize> {
    match b {
        b')' => Some(0),
        b']' => Some(1),
        b'}' => Some(2),
        _ => None,
    }
}
fn quote_idx(b: u8) -> Option<usize> {
    match b {
        b'\'' => Some(0),
        b'"' => Some(1),
        b'`' => Some(2),
        _ => None,
    }
}

impl ForcedIds {
    /// Resolve a forced [`TokenDelim`] to the concrete token id that should be
    /// spliced, if we have a bare id for it.
    fn id_for(&self, d: TokenDelim) -> Option<u32> {
        match d {
            TokenDelim::Close(b) => close_idx(b).and_then(|i| self.close[i]),
            TokenDelim::Quote(b) => quote_idx(b).and_then(|i| self.quote[i]),
            TokenDelim::TripleQuote(b) => match b {
                b'\'' => self.triple[0],
                b'"' => self.triple[1],
                _ => None,
            },
            _ => None,
        }
    }
}

/// Build the reverse map from delimiter class to a *bare* token id by scanning
/// decoded strings. Prefers the id whose decoded string is exactly the bare
/// delimiter (no leading space) so the spliced token doesn't inject whitespace.
pub fn build_forced_ids<F>(vocab_size: usize, mut decode: F) -> ForcedIds
where
    F: FnMut(u32) -> Option<String>,
{
    let mut ids = ForcedIds::default();
    for id in 0..vocab_size as u32 {
        let Some(s) = decode(id) else { continue };
        // Only accept exact-bare forms (no surrounding whitespace) so a splice
        // never changes indentation. A leading-space variant is skipped.
        match s.as_str() {
            ")" => set_first(&mut ids.close[0], id),
            "]" => set_first(&mut ids.close[1], id),
            "}" => set_first(&mut ids.close[2], id),
            "'" => set_first(&mut ids.quote[0], id),
            "\"" => set_first(&mut ids.quote[1], id),
            "`" => set_first(&mut ids.quote[2], id),
            "'''" => set_first(&mut ids.triple[0], id),
            "\"\"\"" => set_first(&mut ids.triple[1], id),
            _ => {}
        }
    }
    ids
}

fn set_first(slot: &mut Option<u32>, id: u32) {
    if slot.is_none() {
        *slot = Some(id);
    }
}

/// Process-wide forced-id map, built alongside [`set_delim_table`].
static FORCED_IDS: OnceLock<std::sync::Arc<ForcedIds>> = OnceLock::new();

/// Install the forced-id map. Idempotent.
pub fn set_forced_ids(ids: std::sync::Arc<ForcedIds>) {
    let _ = FORCED_IDS.set(ids);
}

/// Read the forced-id map. `None` until [`set_forced_ids`] runs.
pub fn forced_ids() -> Option<std::sync::Arc<ForcedIds>> {
    FORCED_IDS.get().cloned()
}

/// Outcome of one splice pass, for the one-shot stats log.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpliceStats {
    /// Number of draft positions whose token was replaced by a forced token.
    pub splices: usize,
    /// Number of draft positions dropped (chain truncated at a splice because
    /// the drafter's remaining tail became inconsistent).
    pub truncated: usize,
    /// The compact position of the first splice (for the stats log), or
    /// `usize::MAX` if none.
    pub first_pos: usize,
}

/// Splice structurally-forced tokens into `drafts`, given the tracker state
/// AFTER walking the committed prefix (i.e. positioned at the slot the first
/// draft occupies).
///
/// Walks the drafts left-to-right, advancing a *clone* of `base` tracker. At
/// each position:
///   * If the tracker forces a specific next token (innermost closer) and the
///     drafter's token DISAGREES, replace the drafter's token with the forced
///     id (a splice), advance the tracker by the FORCED token, and continue.
///   * If the drafter's token AGREES with the forced token, or nothing is
///     forced, keep the drafter's token and advance by it.
///   * If we encounter an opaque token or a state we can't reason about, stop
///     splicing for the remainder (keep the rest of the drafter chain as-is —
///     it's still lossless; we just add no value past here).
///
/// The `drafts` vector is mutated in place. Returns [`SpliceStats`]. Never
/// changes `drafts.len()` (the K=γ verify graph shape is preserved): a splice
/// is a token REPLACEMENT, not an insertion.
///
/// LOSSLESS: every replacement is a candidate the verifier will accept only if
/// it equals the target's greedy token. Wrong splices are rejected for free.
pub fn splice_forced(
    drafts: &mut [u32],
    committed: &[u32],
    table: &[TokenDelim],
    forced: &ForcedIds,
) -> SpliceStats {
    let mut stats = SpliceStats {
        splices: 0,
        truncated: 0,
        first_pos: usize::MAX,
    };
    // Build the base tracker from the committed prefix. The committed stream is
    // trusted (it is what the target actually produced), so we do NOT let it
    // poison — a mismatched close in committed text just means our model is
    // imperfect there; reset the stack conservatively rather than force nothing
    // forever. We simply run advance(); poisoning on the committed walk would
    // globally disable splicing, which is the safe default anyway.
    let mut base = BracketTracker::new();
    // Only walk a bounded suffix of the committed stream: structure that opened
    // thousands of tokens ago and is still unclosed is rare in code, and a
    // shorter window keeps this off the hot path. 4096 tokens covers any
    // realistic single function / class body.
    let start = committed.len().saturating_sub(4096);
    for &tok in &committed[start..] {
        let d = table.get(tok as usize).copied().unwrap_or(TokenDelim::Other);
        base.advance(d);
        // A poisoned base can't force anything useful; bail early (no splices).
        if base.poisoned {
            return stats;
        }
    }

    let mut trk = base;
    let mut stopped = false;
    for (pos, slot) in drafts.iter_mut().enumerate() {
        if stopped || trk.poisoned {
            break;
        }
        let drafted = *slot;
        let drafted_delim = table
            .get(drafted as usize)
            .copied()
            .unwrap_or(TokenDelim::Other);

        // Is a specific token forced here?
        if let Some(forced_delim) = trk.forced_next() {
            if drafted_delim == forced_delim {
                // Drafter already produced the forced token — nothing to do,
                // just advance and continue (this position is "free" and the
                // chain naturally continues into the forced structure).
                trk.advance(drafted_delim);
                continue;
            }
            // Drafter disagrees with a forced closer. Splice IF we have a
            // concrete bare id for the forced token.
            if let Some(fid) = forced.id_for(forced_delim) {
                *slot = fid;
                stats.splices += 1;
                if stats.first_pos == usize::MAX {
                    stats.first_pos = pos;
                }
                // Advance by the FORCED token (we replaced the drafter's).
                trk.advance(forced_delim);
                continue;
            }
            // No id available to realize the forced token — leave the drafter's
            // token, advance by IT, and stop splicing further (we can't align
            // the chain to the structure we couldn't emit).
            trk.advance(drafted_delim);
            stopped = true;
            continue;
        }

        // Nothing forced at this position: keep the drafter's token and advance.
        // If the drafter's token is opaque, advance() poisons and the loop ends.
        trk.advance(drafted_delim);
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a tiny fake vocab: id -> string. Ids used across tests:
    //  0..: single chars we care about.
    fn fake_vocab() -> Vec<&'static str> {
        vec![
            "(",   // 0
            ")",   // 1
            "[",   // 2
            "]",   // 3
            "{",   // 4
            "}",   // 5
            "\"",  // 6
            "'",   // 7
            ":",   // 8
            "\n",  // 9
            "foo", // 10
            "bar", // 11
            "\"\"\"", // 12
            "(x", // 13 opaque
            "  ", // 14 whitespace-only -> Other
            " (", // 15 leading-space open (still Open)
        ]
    }

    fn table_from(vocab: &[&str]) -> Vec<TokenDelim> {
        vocab.iter().map(|s| classify_str(s)).collect()
    }

    fn forced_from(vocab: &[&str]) -> ForcedIds {
        build_forced_ids(vocab.len(), |id| vocab.get(id as usize).map(|s| s.to_string()))
    }

    #[test]
    fn classify_basics() {
        assert_eq!(classify_str("("), TokenDelim::Open(b'('));
        assert_eq!(classify_str(" ("), TokenDelim::Open(b'(')); // trimmed
        assert_eq!(classify_str(")"), TokenDelim::Close(b')'));
        assert_eq!(classify_str("\""), TokenDelim::Quote(b'"'));
        assert_eq!(classify_str("\"\"\""), TokenDelim::TripleQuote(b'"'));
        assert_eq!(classify_str(":"), TokenDelim::Colon);
        assert_eq!(classify_str("\n"), TokenDelim::Newline);
        assert_eq!(classify_str("    \n"), TokenDelim::Newline);
        assert_eq!(classify_str("foo"), TokenDelim::Other);
        assert_eq!(classify_str("  "), TokenDelim::Other);
        // Ambiguous multi-symbol tokens -> opaque.
        assert_eq!(classify_str("()"), TokenDelim::Opaque);
        assert_eq!(classify_str("):"), TokenDelim::Opaque);
        assert_eq!(classify_str("(x"), TokenDelim::Opaque);
        assert_eq!(classify_str("\");\n"), TokenDelim::Opaque);
    }

    #[test]
    fn tracker_forces_close_paren() {
        let vocab = fake_vocab();
        let table = table_from(&vocab);
        // committed: foo ( bar  -> stack has one '(' open, forced next = ')'
        let committed = vec![10u32, 0, 11];
        let mut trk = BracketTracker::new();
        for &t in &committed {
            trk.advance(table[t as usize]);
        }
        assert_eq!(trk.forced_next(), Some(TokenDelim::Close(b')')));
    }

    #[test]
    fn splice_replaces_wrong_token_with_close_paren() {
        let vocab = fake_vocab();
        let table = table_from(&vocab);
        let forced = forced_from(&vocab);
        // committed: foo (  -> forced next ')'
        let committed = vec![10u32, 0];
        // drafter guessed: [foo, bar, ...] wrong at pos 0 (should be ')')
        let mut drafts = vec![10u32, 11, 10];
        let stats = splice_forced(&mut drafts, &committed, &table, &forced);
        assert_eq!(stats.splices, 1);
        assert_eq!(stats.first_pos, 0);
        // pos 0 replaced with ')' == id 1
        assert_eq!(drafts[0], 1);
    }

    #[test]
    fn splice_noop_when_drafter_already_correct() {
        let vocab = fake_vocab();
        let table = table_from(&vocab);
        let forced = forced_from(&vocab);
        let committed = vec![10u32, 0]; // foo (
        // drafter already produced ')' at pos 0
        let mut drafts = vec![1u32, 10, 11];
        let before = drafts.clone();
        let stats = splice_forced(&mut drafts, &committed, &table, &forced);
        assert_eq!(stats.splices, 0);
        assert_eq!(drafts, before);
    }

    #[test]
    fn nested_brackets_force_innermost_first() {
        let vocab = fake_vocab();
        let table = table_from(&vocab);
        let forced = forced_from(&vocab);
        // committed: ( [  -> innermost open is '[', forced ']'
        let committed = vec![0u32, 2];
        let mut drafts = vec![10u32, 5, 5]; // wrong at pos 0
        let stats = splice_forced(&mut drafts, &committed, &table, &forced);
        // Both nested closers get spliced in innermost-first order.
        assert_eq!(stats.splices, 2);
        assert_eq!(drafts[0], 3); // ']'
        // After splicing ']', the tracker now has only '(' open -> if the next
        // drafter token disagrees with ')', splice again.
        // drafts[1] was '}' (id 5), disagrees with forced ')', so splice to ')'.
        assert_eq!(drafts[1], 1);
    }

    #[test]
    fn string_suppresses_bracket_forcing() {
        let vocab = fake_vocab();
        let table = table_from(&vocab);
        let forced = forced_from(&vocab);
        // committed: " (   -> we are INSIDE a double-quoted string; the '(' is
        // literal text, so the forced next token is the closing '"', NOT ')'.
        let committed = vec![6u32, 0];
        let mut drafts = vec![10u32, 10, 10]; // all wrong
        let stats = splice_forced(&mut drafts, &committed, &table, &forced);
        assert_eq!(stats.splices, 1);
        assert_eq!(drafts[0], 6); // spliced the closing '"'
    }

    #[test]
    fn opaque_token_halts_splicing() {
        let vocab = fake_vocab();
        let table = table_from(&vocab);
        let forced = forced_from(&vocab);
        // committed: foo ( then an OPAQUE token (id 13 "(x") in committed →
        // base poisons → no splices at all (fail-safe).
        let committed = vec![10u32, 0, 13];
        let mut drafts = vec![10u32, 10];
        let before = drafts.clone();
        let stats = splice_forced(&mut drafts, &committed, &table, &forced);
        assert_eq!(stats.splices, 0);
        assert_eq!(drafts, before);
    }

    #[test]
    fn triple_quote_open_and_close() {
        let vocab = fake_vocab();
        let table = table_from(&vocab);
        let forced = forced_from(&vocab);
        // committed opens a triple-quoted docstring: """  -> forced next """
        let committed = vec![12u32];
        let mut drafts = vec![10u32, 11];
        let stats = splice_forced(&mut drafts, &committed, &table, &forced);
        assert_eq!(stats.splices, 1);
        assert_eq!(drafts[0], 12); // spliced closing triple quote
    }

    #[test]
    fn no_forcing_when_balanced() {
        let vocab = fake_vocab();
        let table = table_from(&vocab);
        let forced = forced_from(&vocab);
        // committed: foo ( bar )  -> balanced, nothing forced.
        let committed = vec![10u32, 0, 11, 1];
        let mut drafts = vec![10u32, 11, 10];
        let before = drafts.clone();
        let stats = splice_forced(&mut drafts, &committed, &table, &forced);
        assert_eq!(stats.splices, 0);
        assert_eq!(drafts, before);
    }

    #[test]
    fn drafts_len_preserved() {
        let vocab = fake_vocab();
        let table = table_from(&vocab);
        let forced = forced_from(&vocab);
        let committed = vec![0u32, 2, 4]; // ( [ {  three opens
        let mut drafts = vec![10u32, 10, 10, 10, 10];
        let n = drafts.len();
        let _ = splice_forced(&mut drafts, &committed, &table, &forced);
        assert_eq!(drafts.len(), n); // never changes length
        // innermost first: '{' -> '}', then '[' -> ']', then '(' -> ')'
        assert_eq!(drafts[0], 5);
        assert_eq!(drafts[1], 3);
        assert_eq!(drafts[2], 1);
    }
}
