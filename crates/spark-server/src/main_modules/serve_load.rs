//! Process-scoped state that outlives any single model.
//!
//! PORTED SHAPE, NOT PORTED CODE. Upstream's `serve_load.rs` is ~1350 lines: it
//! loads a checkpoint and builds everything derived from it. That whole path is
//! the model-swap feature, which this tree deliberately does not have (see the
//! toast in `tui::lib_state::launch`, which names `model_swap`, this `Carried`,
//! and a `startup()` that returns the scheduler handle).
//!
//! What IS needed here is only the small part the dashboard and the request
//! handlers touch: the three stores that must survive a swap. `model_host` holds
//! one of these and hands out `rate_limiter()` while no model is loaded, so the
//! type has to exist even though nothing constructs a second model yet.
//!
//! Rebuilding these on a swap silently drops stored conversations and responses
//! and resets every rate-limit bucket — no error, just a user noticing their
//! history is gone. Keeping them in one struct makes carrying them the only way
//! to load a second model: the compiler asks for them, so nobody has to remember.

use std::sync::Arc;

use crate::main_modules::AppState;
use crate::{conversation_store, rate_limiter, response_store};

/// State that OUTLIVES any model and must be carried across a swap.
#[derive(Clone)]
pub(crate) struct Carried {
    pub response_store: Arc<response_store::ResponseStore>,
    pub rate_limiter: Arc<rate_limiter::RateLimiter>,
    pub conversation_store: Arc<conversation_store::ConversationStore>,
}

impl Carried {
    /// First boot: build them once, from the environment.
    ///
    /// Installed on the HOST before the listener binds, so everything
    /// process-scoped is reachable while no model is loaded — and so there is
    /// exactly one of each: handlers refund through the same limiter the
    /// middleware debits, and read the same stores a swap carries forward.
    ///
    /// # Errors
    /// None today. The `Result` matches upstream's signature deliberately:
    /// there, a malformed `*_RATE_LIMIT_*` or `*_STORE_*` value is REFUSED
    /// rather than defaulted, because the default for a rate limit is "no
    /// limit" and a swallowed typo opens the server up silently. Our three
    /// constructors are currently infallible, so this cannot fail yet; keeping
    /// the fallible shape means adding a validating parser later is a change to
    /// one function rather than to every call site.
    pub fn from_env() -> Result<Self, String> {
        Ok(Self {
            response_store: response_store::ResponseStore::from_env(),
            rate_limiter: rate_limiter::RateLimiter::from_env(),
            conversation_store: conversation_store::ConversationStore::from_env(),
        })
    }

    /// A swap: take them from the model being replaced.
    ///
    /// Unused while this tree has no swap path, and kept because it is the
    /// half of the contract that makes the other half meaningful — the point
    /// of `Carried` is that the second model gets THESE instances, not new ones.
    #[allow(dead_code)]
    pub fn from_previous(previous: &AppState) -> Self {
        Self {
            response_store: previous.response_store.clone(),
            rate_limiter: previous.rate_limiter.clone(),
            conversation_store: previous.conversation_store.clone(),
        }
    }
}
