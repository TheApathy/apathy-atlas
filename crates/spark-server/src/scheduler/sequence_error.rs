// SPDX-License-Identifier: AGPL-3.0-only

use super::ActiveSeq;

/// Mark, but do not free: the ordinary retirement owner sends one error and
/// releases the sequence exactly once. Never turn a failed verifier into Done.
pub(super) fn mark_sequence_error(a: &mut ActiveSeq, stage: &str, error: &anyhow::Error) {
    tracing::error!(stage, error = %error, "inference sequence failed");
    a.terminal_error
        .get_or_insert_with(|| format!("{stage}: {error:#}"));
    a.pending_drafts.clear();
    a.finished = true;
}
