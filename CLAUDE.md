# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`ltsis` is a Rust library implementing O(n) multinomial sampling for sequential importance sampling (Massey, ICASSP 2008). `f32` throughout, allocation-free, `no_std`-compatible. Real deployment target is Cortex-M4F (single-precision FPU).

## Commands

- Build: `cargo build` (default `std`) or `cargo build --no-default-features --features libm` (no_std)
- Tests: `cargo test --release` — statistical tests are slow in debug; always use `--release`
- Single test: `cargo test --release <test_name>` (e.g. `ks_against_theory`)
- Benchmark: `cargo bench` (custom `harness = false` main, not `#[bench]`)
- Example: `cargo run --example quick_start --release`
- Docs: `cargo doc --all-features --open`

## Feature flags

Exactly one of `std` (default) or `libm` must be enabled — both supply `powf`/`Exp1` math. Tests and benches require `std`. The crate deliberately does **not** enable `rand/std_rng`; tests/benches/example use `SmallRng` (Xoshiro256++) to avoid the ChaCha20 dep chain.

## Architecture

Everything lives in `src/lib.rs` (intentionally flat — the four-file
split was excessive for a crate this small). The file is organized
top-to-bottom as:

- `first_uniform(rng, k)` — samples `min(U₁..Uₖ) ~ Beta(1, k)` via inverse CDF `1 − (1 − u)^(1/k)`. The `1 − (1 − u)^…` form (not `1 − u^…`) is load-bearing for f32 boundary safety; see INTERNALS §4.1.
- `SortedUniforms` iterator — uses the Bentley–Saxe spacings recurrence on top of `first_uniform`. One `powf` per element.
- `sample_indices` (streaming) — uses `SortedUniforms` to merge against the cumulative weight array.
- `sample_indices_buffered` — avoids `powf` via the Gamma-ratio identity, stashing intermediate `Exp(1)` draws in `out` via `f32::to_bits`/`from_bits` round-trip — the `out: &mut [u32]` slot doubles as f32 scratch. Indices are `u32` (not `usize`) to make this work portably.

## Invariants — read INTERNALS.md before changing the merge or summation

Memory safety of the merge depends on **Lemma 3** (INTERNALS §5.2): the up-front `total = Σ weights` and the merge's incremental `cumulative_w` must end bit-identical when the cursor reaches the last weight. This requires both walks to:

- Use **Kahan compensated summation** (`kahan_add` in `lib.rs`).
- Walk `weights` in increasing index order from `(sum, c) = (0, 0)`.
- Use the same algorithm — no parallel reduce, no SIMD reduction, no swapping in a different accumulator.

Breaking any of these invalidates Lemma 3 and the merge can index out of bounds. The buffered variant additionally needs the `target.min(total)` clip at the right endpoint.

## Statistical tests

`tests/statistical.rs` runs ten KS / χ² / moment tests with **fixed RNG seeds**, calibrated so the aggregate random-failure probability under correct code is < 1e-9. `Cargo.lock` is committed because the seed budget assumes a specific `rand`/`rand_distr` algorithm; if those bump variate-generation algorithms (or our own `rand` usage shifts), seeds produce different sequences and thresholds may need re-derivation. See INTERNALS §5.4.

## Documentation conventions

- Markdown line length ≤ 65 chars where reasonable (matches existing files).
- Rustdoc examples must compile; show full setup including imports.
- Authorship metadata in `Cargo.toml` lists Claude as an author. Per user policy: Claude goes in `authors`, never on a `Copyright` line.

## Out-of-repo files

`HANDOFF.md` for this project lives at `~/.claude/projects/-usr-local-src-ltsis/HANDOFF.md`, not in the repo (no `exclude` in `Cargo.toml`).
