// SPDX-License-Identifier: AGPL-3.0-only

//! The Benchmarks pane: is the box fit to measure on, and what did the
//! harnesses record?
//!
//! Upstream's tab RAN benchmarks through `avarok-plugin`. This one does not
//! run anything, deliberately: the measurements that matter on this box are
//! taken by the `bench/` harnesses, whose controls, prewarm assertions and
//! lock discipline are what make their numbers trustworthy. A second runner
//! in the dashboard would produce numbers that look the same and are not
//! comparable, which is worse than having no tab.
//!
//! So it answers the two questions asked either side of a run: may I measure
//! now, and what did the last ones say?

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph};

use crate::tui::app::App;
use crate::tui::app::BenchSub;
use crate::tui::data::bench;
use crate::tui::theme;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    match app.bench_sub {
        BenchSub::Box => draw_box(f, area),
        BenchSub::Runs => draw_runs(f, app, area),
    }
}

fn block(title: &str) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::border(false))
        .title(format!(" {title} "))
}

fn draw_box(f: &mut Frame, area: Rect) {
    let st = bench::box_state(std::path::Path::new(
        &std::env::var("ATLAS_GPU_LOCK").unwrap_or_else(|_| "/home/flocka/atlas/.gb10.lock".into()),
    ));
    let verdict = st.verdict();
    let blocked = verdict.starts_with("DO NOT");
    let mut lines = vec![
        Line::from(Span::styled(
            verdict,
            Style::default()
                .fg(if blocked {
                    theme::ERROR.color()
                } else {
                    theme::GREEN.color()
                })
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(format!(
            "  GPU lock     {}",
            match st.lock_holder {
                Some(p) => format!("held by pid {p}"),
                None => "free".into(),
            }
        )),
        Line::from(format!(
            "  queued       {}",
            if st.waiters.is_empty() {
                "nobody".to_string()
            } else {
                st.waiters
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        )),
        Line::from(format!(
            "  host memory  {:.1} GB available",
            st.mem_available_mib as f64 / 1024.0
        )),
        Line::from(format!("  builders     {}", st.builders)),
        Line::from(""),
        // The three things that have actually invalidated measurements here,
        // named so the pane teaches the hazard rather than only reporting it.
        Line::from(Span::styled(
            "  A run taken while another job holds the lock is not a slower",
            theme::dim(),
        )),
        Line::from(Span::styled(
            "  number — it is a wrong one. `nvidia-smi` shows a clear GPU",
            theme::dim(),
        )),
        Line::from(Span::styled(
            "  between arms while the lock is correctly held: the lock is the",
            theme::dim(),
        )),
        Line::from(Span::styled(
            "  authority, the GPU is a lagging indicator.",
            theme::dim(),
        )),
    ];
    if st.lock_holder.is_some() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  A dead holder pid is NOT a stale lock: it survives through a",
            theme::dim(),
        )));
        lines.push(Line::from(Span::styled(
            "  descriptor an alive child inherited. Do not clear it.",
            theme::dim(),
        )));
    }
    f.render_widget(Paragraph::new(lines).block(block("Box readiness")), area);
}

fn draw_runs(f: &mut Frame, app: &App, area: Rect) {
    let dirs = bench::runs_dirs();
    if dirs.is_empty() {
        f.render_widget(
            Paragraph::new(vec![
                Line::from("  No run directories configured."),
                Line::from(""),
                Line::from(Span::styled(
                    "  Set ATLAS_BENCH_RUNS to a colon-separated list of the",
                    theme::dim(),
                )),
                Line::from(Span::styled(
                    "  harnesses' runs/ directories, e.g.",
                    theme::dim(),
                )),
                Line::from(Span::styled(
                    "    ATLAS_BENCH_RUNS=/home/flocka/atlas/glm53-prefill-work/bench/runs",
                    theme::dim(),
                )),
            ])
            .block(block("Recorded runs")),
            area,
        );
        return;
    }

    let mut arms: Vec<bench::ArmSummary> = Vec::new();
    for d in &dirs {
        arms.extend(bench::scan_runs(d));
    }

    // Arms whose name marks them a control. The spread of THESE is the floor
    // under every delta on that harness — the single most useful number this
    // campaign produced, and one nobody computed for three sweeps.
    let controls: Vec<bench::ArmSummary> = arms
        .iter()
        .filter(|a| {
            let n = a.name.to_ascii_lowercase();
            n.contains("-c") || n.contains("ctrl") || n.contains("control")
        })
        .cloned()
        .collect();

    let mut lines = Vec::new();
    match bench::control_spread_pct(&controls) {
        Some(pct) => lines.push(Line::from(Span::styled(
            format!(
                "  control spread {pct:.2}%  over {} arms — THE FLOOR: a delta \
                 smaller than this is invisible, not small",
                controls.len()
            ),
            Style::default()
                .fg(theme::WARN.color())
                .add_modifier(Modifier::BOLD),
        ))),
        None => lines.push(Line::from(Span::styled(
            "  fewer than two control arms — no floor can be computed, and one \
             arm reported as 0% would be a two-sample underestimate",
            theme::dim(),
        ))),
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!(
            "  {:<22} {:>6} {:>11} {:>10} {:>10}",
            "arm", "trials", "median tok/s", "min", "max"
        ),
        theme::dim(),
    )));
    for a in arms.iter().take(area.height.saturating_sub(6) as usize) {
        lines.push(Line::from(format!(
            "  {:<22} {:>6} {:>11.1} {:>10.1} {:>10.1}",
            a.name, a.trials, a.median_tok_s, a.min_tok_s, a.max_tok_s
        )));
    }
    let _ = app;
    f.render_widget(Paragraph::new(lines).block(block("Recorded runs")), area);
}

/// Split helper kept for symmetry with the other tabs.
#[allow(dead_code)]
fn halves(area: Rect) -> Vec<Rect> {
    Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area)
        .to_vec()
}
