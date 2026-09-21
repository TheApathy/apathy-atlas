// SPDX-License-Identifier: AGPL-3.0-only

//! Render smoke tests over `TestBackend`.
//!
//! Layout code is where a TUI actually crashes: a `Rect` computed past the
//! frame, a `split` with more constraints than cells, a subtraction that
//! underflows on a narrow terminal. None of that is visible to `cargo check`,
//! and all of it takes the dashboard — and with it the server's foreground —
//! down at runtime. Rendering every section into a buffer at several sizes is
//! the cheapest thing that catches it.

use ratatui::Terminal;
use ratatui::backend::TestBackend;

use super::draw;
use crate::tui::app::{App, Section};

pub(super) fn app() -> App {
    use clap::Parser;
    let app = App::new(crate::cli::ServeArgs::parse_from([
        "spark",
        "nvidia/Qwen3.6-27B-NVFP4",
    ]));
    app
}

pub(super) fn render(app: &App, width: u16, height: u16) -> String {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("backend");
    terminal.draw(|f| draw(f, app)).expect("draw");
    terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol())
        .collect()
}

/// The sizes that matter: the wide layout, the narrow-sidebar layout
/// (width < 96), the short-header layout (height < 28), and a terminal small
/// enough that every `saturating_sub` in the tree is exercised.
const SIZES: [(u16, u16); 4] = [(160, 48), (100, 30), (80, 24), (40, 12)];

#[test]
fn every_section_renders_at_every_size() {
    for section in Section::ALL {
        for (w, h) in SIZES {
            let mut a = app();
            a.section = section;
            let out = render(&a, w, h);
            assert!(
                !out.is_empty(),
                "{} at {w}x{h} drew nothing",
                section.label()
            );
        }
    }
}

#[test]
fn a_terminal_one_cell_wide_does_not_panic() {
    // Underflow guard: every layout in the tree subtracts from the width.
    for (w, h) in [(1, 1), (2, 3), (1, 40), (40, 1)] {
        let mut a = app();
        a.section = Section::Library;
        let _ = render(&a, w, h);
    }
}

/// The Library panes must render at realistic and hostile sizes.
mod library {
    use super::*;
    use crate::tui::lib_state::View as LibView;

    fn with_rows() -> App {
        let mut app = app();
        app.section = Section::Library;
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/recipes/qwen3.6/qwen3.6-35b-a3b-fp8-mtp.yaml");
        let recipe = crate::recipe::Recipe::parse(
            "qwen3.6/flagship",
            &std::fs::read_to_string(path).expect("fixture"),
        )
        .expect("parses");
        app.library = vec![crate::tui::data::library::LibraryEntry {
            id: recipe.model.clone(),
            snapshot_dir: Default::default(),
            size_bytes: 34_900_000_000,
            has_weights: true,
            model_type: "qwen3_6_moe".into(),
            quant: "fp8".into(),
            layers: 40,
            hidden: 4096,
            heads: 32,
            experts: 128,
            context: 65536,
            optimized: true,
        }];
        app.lib.index = crate::recipe::fetch::Index {
            recipes: vec![recipe],
            ..Default::default()
        };
        app.lib.rebuild(&app.library);
        app
    }

    /// The defect this test exists for: the panel title reported a row count
    /// that disagreed with the rows drawn beneath it, because the count and the
    /// list were read from different places.
    #[test]
    fn the_title_agrees_with_the_rows_it_draws() {
        let app = with_rows();
        let out = render(&app, 200, 50);
        assert!(out.contains("MODELS"), "the panel is drawn");
        assert!(
            !out.contains("MODELS ─ 0"),
            "a populated list must not claim 0 rows:\n{out}"
        );
        assert!(
            !out.contains("no models or recipes yet"),
            "the empty hint must not appear beside real rows:\n{out}"
        );
        assert!(out.contains("Qwen3.6-35B-A3B-FP8"), "the row is drawn");
    }

    #[test]
    fn the_empty_state_says_what_to_do() {
        let mut app = app();
        app.section = Section::Library;
        let out = render(&app, 200, 50);
        assert!(out.contains("press r to fetch recipes"), "{out}");
    }

    #[test]
    fn the_config_pane_renders_and_shows_the_command() {
        let mut app = with_rows();
        app.lib.open_cards().expect("opens");
        app.lib.open_config().expect("opens");
        assert_eq!(app.lib.view, LibView::Config);
        let out = render(&app, 200, 50);
        assert!(out.contains("SETTINGS"), "{out}");
        assert!(out.contains("spark serve"), "the command preview: {out}");
    }

    /// The cards pane: the choice between sibling recipes, with the room the
    /// list row never had.
    #[test]
    fn the_cards_pane_shows_the_recipe_and_its_rationale() {
        let mut app = with_rows();
        app.lib.open_cards().expect("opens");
        assert_eq!(app.lib.view, LibView::Cards);
        let out = render(&app, 200, 50);
        assert!(out.contains("recipe"), "the header counts them: {out}");
        // The description is the measured rationale — the reason this pane
        // exists rather than a one-line row.
        assert!(out.contains("FLAGSHIP"), "the recipe's own text: {out}");
        assert!(out.contains("configure and start"), "{out}");
    }

    /// A one-recipe model still gets a card, by explicit request.
    #[test]
    fn one_recipe_still_renders_a_card() {
        let mut app = with_rows();
        assert_eq!(app.lib.cards().len(), 1, "the fixture has one");
        app.lib.open_cards().expect("opens");
        let out = render(&app, 200, 50);
        assert!(
            out.contains("1 recipe"),
            "singular, not \"1 recipes\": {out}"
        );
    }

    /// Narrow and short terminals are where layout maths underflows.
    #[test]
    fn the_library_survives_hostile_sizes() {
        let app = with_rows();
        for (w, h) in [(40, 12), (60, 20), (80, 24), (120, 30), (240, 80)] {
            let _ = render(&app, w, h);
        }
        let mut cards = with_rows();
        cards.lib.open_cards().expect("opens");
        for (w, h) in [(40, 12), (60, 20), (80, 24), (240, 80)] {
            let _ = render(&cards, w, h);
        }
        let mut config = with_rows();
        config.lib.open_cards().expect("opens");
        config.lib.open_config().expect("opens");
        for (w, h) in [(40, 12), (60, 20), (80, 24), (240, 80)] {
            let _ = render(&config, w, h);
        }
    }
}

/// Frames are drawn into a LIVE terminal, one after another, not into a fresh
/// buffer each time. The Library's first frame has no rows (the local scan and
/// the recipe cache both land a tick later), so the empty state is genuinely
/// shown and then replaced — and a stale title left behind by that transition
/// is exactly what a single-frame test cannot see.
#[test]
fn the_library_leaves_nothing_behind_when_it_fills_in() {
    let mut terminal = Terminal::new(TestBackend::new(200, 50)).expect("backend");

    // Frame 1: empty, as on first entry.
    let mut app = app();
    app.section = Section::Library;
    terminal.draw(|f| draw(f, &app)).expect("draw");

    // Frame 2: populated, as one tick later.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/recipes/qwen3.6/qwen3.6-35b-a3b-fp8-mtp.yaml");
    let recipe = crate::recipe::Recipe::parse(
        "qwen3.6/flagship",
        &std::fs::read_to_string(path).expect("fixture"),
    )
    .expect("parses");
    app.lib.index = crate::recipe::fetch::Index {
        recipes: vec![recipe],
        ..Default::default()
    };
    app.lib.rebuild(&[]);
    terminal.draw(|f| draw(f, &app)).expect("draw");

    let out: String = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol())
        .collect();
    assert!(
        !out.contains("no models or recipes yet"),
        "the empty hint survived into the populated frame:\n{out}"
    );
    assert!(
        !out.contains("MODELS ─ 0"),
        "the empty title survived into the populated frame:\n{out}"
    );
    assert!(out.contains("MODELS ─ 1"), "the new title is drawn:\n{out}");
}

#[test]
fn the_clear_chat_prompt_names_what_it_will_discard() {
    use crate::tui::app::TermSub;
    use crate::tui::chat::{ChatMessage, Role};
    let mut a = app();
    a.section = Section::Terminal;
    a.term_sub = TermSub::Chat;
    a.chat
        .transcript
        .push(ChatMessage::new(Role::User, "hello".into()));
    a.chat
        .transcript
        .push(ChatMessage::new(Role::Model, "hi".into()));
    a.confirm_chat_clear = true;
    let out = render(&a, 120, 40);
    assert!(out.contains("CLEAR THE CONVERSATION?"), "{out}");
    assert!(
        out.contains("2 turns will be discarded"),
        "the stake is named in the user's own units:\n{out}"
    );
    assert!(
        out.contains("any other key"),
        "the way out is named:\n{out}"
    );
}

#[test]
fn the_new_chat_key_is_discoverable_from_the_chat_pane() {
    use crate::tui::app::TermSub;
    let mut a = app();
    a.section = Section::Terminal;
    a.term_sub = TermSub::Chat;
    let out = render(&a, 160, 48);
    assert!(
        out.contains("Ctrl+N new chat"),
        "a reset nobody can find is no reset:\n{out}"
    );
}
