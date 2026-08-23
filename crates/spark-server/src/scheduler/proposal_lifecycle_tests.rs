// SPDX-License-Identifier: AGPL-3.0-only

use std::cell::{Cell, RefCell};

use super::*;

#[derive(Debug, PartialEq, Eq)]
struct Tree(u8);

#[test]
fn proposal_results_drain_once_and_replace_the_whole_frame() {
    for (result, expected, fails) in [
        (Ok(vec![10, 11]), (vec![10, 11], Some(Tree(2))), false),
        (Ok(Vec::new()), (Vec::new(), None), false),
        (Err("failed"), (Vec::new(), None), true),
    ] {
        let drains = Cell::new(0);
        let mut drafts = vec![1, 2];
        let mut tree = Some(Tree(1));
        let outcome = install(&mut drafts, &mut tree, result, || {
            drains.set(drains.get() + 1);
            Some(Tree(2))
        });
        assert_eq!(outcome.is_err(), fails);
        assert_eq!(drains.get(), 1);
        assert_eq!((drafts, tree), expected);
    }
}

#[test]
fn flat_or_pld_result_replaces_stale_tree() {
    let mut drafts = vec![1];
    let mut tree = Some(Tree(7));
    install(&mut drafts, &mut tree, Ok::<_, ()>(vec![20, 21]), || None).unwrap();
    assert_eq!(drafts, [20, 21]);
    assert_eq!(tree, None);
}

#[test]
fn width_one_and_two_counterexamples_cannot_mix_old_and_new_frames() {
    for width in [1, 2] {
        let mut drafts = vec![10; width];
        let mut tree = Some(Tree(1));
        install(&mut drafts, &mut tree, Ok::<_, ()>(vec![20; width]), || {
            Some(Tree(2))
        })
        .unwrap();
        assert_eq!(drafts, vec![20; width]);
        assert_eq!(tree, Some(Tree(2)));

        install(&mut drafts, &mut tree, Ok::<_, ()>(Vec::new()), || {
            Some(Tree(3))
        })
        .unwrap();
        assert!(drafts.is_empty());
        assert_eq!(tree, None);
    }
}

#[test]
fn batched_success_empty_and_error_replace_each_frame_independently() {
    let mut frames = [
        (vec![1], Some(Tree(1))),
        (vec![2], Some(Tree(2))),
        (vec![3], Some(Tree(3))),
    ];
    let drains = [Cell::new(0), Cell::new(0), Cell::new(0)];

    install(
        &mut frames[0].0,
        &mut frames[0].1,
        Ok::<_, &str>(vec![10]),
        || {
            drains[0].set(drains[0].get() + 1);
            Some(Tree(10))
        },
    )
    .unwrap();
    install(
        &mut frames[1].0,
        &mut frames[1].1,
        Ok::<_, &str>(Vec::new()),
        || {
            drains[1].set(drains[1].get() + 1);
            Some(Tree(11))
        },
    )
    .unwrap();
    assert!(
        install(&mut frames[2].0, &mut frames[2].1, Err("failed"), || {
            drains[2].set(drains[2].get() + 1);
            Some(Tree(12))
        })
        .is_err()
    );

    assert_eq!(frames[0], (vec![10], Some(Tree(10))));
    assert_eq!(frames[1], (Vec::new(), None));
    assert_eq!(frames[2], (Vec::new(), None));
    assert!(drains.iter().all(|count| count.get() == 1));
}

#[test]
fn flat_take_refuses_topology_without_consuming_either_half() {
    let mut drafts = vec![1, 2];
    let tree = Some(Tree(4));
    assert_eq!(take_flat(&mut drafts, &tree), None);
    assert_eq!(drafts, [1, 2]);

    let tree = None::<Tree>;
    assert_eq!(take_flat(&mut drafts, &tree), Some(vec![1, 2]));
    assert!(drafts.is_empty());
}

#[test]
fn flat_batch_take_leaves_no_topology_for_terminal_paths() {
    for width in [1, 2] {
        let mut drafts = vec![9; width];
        let tree = None::<Tree>;
        let consumed = take_flat(&mut drafts, &tree).unwrap();
        assert_eq!(consumed, vec![9; width]);
        assert!(drafts.is_empty());
        assert_eq!(tree, None);
    }
}

#[test]
fn prefix_lifecycle_preserves_only_an_unchanged_matching_frame() {
    let mut drafts = vec![1, 2, 3];
    let mut tree = Some(Tree(5));
    assert!(retain_prefix(&mut drafts, &mut tree, 3));
    assert_eq!(tree, Some(Tree(5)));

    assert!(retain_prefix(&mut drafts, &mut tree, 2));
    assert_eq!(drafts, [1, 2]);
    assert_eq!(tree, None);
    tree = Some(Tree(6));
    assert!(!retain_prefix(&mut drafts, &mut tree, 0));
    assert!(drafts.is_empty());
    assert_eq!(tree, None);
}

#[test]
fn invalid_prefix_clears_pair_but_nonempty_frame_is_not_an_orphan() {
    let mut drafts = vec![1];
    let mut tree = Some(Tree(8));
    clear_orphan_tree(&drafts, &mut tree);
    assert_eq!(tree, Some(Tree(8)));

    assert!(!retain_prefix(&mut drafts, &mut tree, 2));
    assert!(drafts.is_empty());
    assert_eq!(tree, None);
}

struct FakeProposalSource {
    proposal: RefCell<Option<anyhow::Result<Vec<u32>>>>,
    tree: RefCell<Option<Tree>>,
    proposals: Cell<usize>,
    takes: Cell<usize>,
}

impl ProposalTreeSource<(), Tree> for FakeProposalSource {
    fn propose(
        &self,
        _state: &mut (),
        _token: u32,
        _position: usize,
        _num_drafts: usize,
        _grammar_bitmask: Option<&[i32]>,
    ) -> anyhow::Result<Vec<u32>> {
        self.proposals.set(self.proposals.get() + 1);
        self.proposal.borrow_mut().take().expect("one proposal")
    }

    fn take_tree(&self, _state: &mut ()) -> Option<Tree> {
        self.takes.set(self.takes.get() + 1);
        self.tree.borrow_mut().take()
    }
}

struct FakeProposalFrame {
    state: (),
    position: usize,
    drafts: Vec<u32>,
    tree: Option<Tree>,
}

impl SchedulerProposalFrame for FakeProposalFrame {
    type State = ();
    type Tree = Tree;

    fn position(&self) -> usize {
        self.position
    }

    fn parts(&mut self) -> (&mut Self::State, &mut Vec<u32>, &mut Option<Self::Tree>) {
        (&mut self.state, &mut self.drafts, &mut self.tree)
    }
}

#[test]
fn production_source_seam_pairs_tree_flat_empty_and_error_with_one_take() {
    let cases = [
        (
            Ok(vec![10, 11]),
            Some(Tree(2)),
            Some(true),
            vec![10, 11],
            Some(Tree(2)),
        ),
        (Ok(vec![20]), None, Some(true), vec![20], None),
        (Ok(Vec::new()), Some(Tree(3)), Some(false), Vec::new(), None),
        (
            Err(anyhow::anyhow!("failed")),
            Some(Tree(4)),
            None,
            Vec::new(),
            None,
        ),
    ];

    for (proposal, offered_tree, expected_outcome, expected_drafts, expected_tree) in cases {
        let source = FakeProposalSource {
            proposal: RefCell::new(Some(proposal)),
            tree: RefCell::new(offered_tree),
            proposals: Cell::new(0),
            takes: Cell::new(0),
        };
        let mut frame = FakeProposalFrame {
            state: (),
            position: 9,
            drafts: vec![1, 2],
            tree: Some(Tree(1)),
        };

        let outcome = propose_and_install(&source, &mut frame, 7, 2, None);

        assert_eq!(outcome.ok(), expected_outcome);
        assert_eq!((frame.drafts, frame.tree), (expected_drafts, expected_tree));
        assert_eq!(source.proposals.get(), 1);
        assert_eq!(source.takes.get(), 1);
    }
}
