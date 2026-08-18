# Apathy Atlas benchmark TUI

The `feat/tui-benchmarks` branch makes benchmark execution a first-class,
inspectable part of the Atlas terminal UI while keeping the benchmark engine
independent of CUDA and model internals.

## Architecture

`atlas-plugin` owns benchmark identity, parameters, execution, cancellation,
artifacts and typed result frames. `spark-server` owns terminal state and
rendering. Plugins emit events over channels; they never mutate TUI state.

The registry currently exposes:

1. concurrency sweep;
2. warm TTFT gate;
3. cold TTFT gate;
4. agentic coding benchmark;
5. BFCL subset;
6. BFCL full.

The executor uses the server's existing Tokio runtime and talks to an
OpenAI-compatible endpoint over HTTP. This keeps client/server behavior in the
measurement while avoiding GPU work in the benchmark crate.

## State-ownership work

The branch grew beyond a UI feature because process-global mutable state made
repeat runs and future hot-swap semantics unsafe. The follow-up commits move
kernel caches and scratch to the backend, model levers and diagnostics to the
model, and sinks/mailboxes/watchdogs to the run. Compile-time descriptor tables
remain static by design.

This distinction is the branch's main engineering result: repeated benchmark
runs are meant to start from explicit run state instead of whatever a previous
model or request left in process globals.

## Validation boundary

Unit and mock-endpoint tests validate the plugin/TUI control plane without a
GPU. A performance claim still requires a model branch's locked live protocol,
output checks and hardware-state record. This branch does not supersede those
model-specific gates.
