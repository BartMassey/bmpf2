# `ltsis` — Sequential Importance Resampling primitives

[![Crates.io](https://img.shields.io/crates/v/ltsis.svg)](https://crates.io/crates/ltsis)
[![Documentation](https://docs.rs/ltsis/badge.svg)](https://docs.rs/ltsis)
[![CI](https://github.com/BartMassey/ltsis/actions/workflows/ci.yml/badge.svg)](https://github.com/BartMassey/ltsis/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/crates/l/ltsis.svg)](#license)

A Rust library for **Sequential Importance Resampling (SIR) with
replacement** — also known as **multinomial resampling** — the
weighted-resampling step at the heart of Bayesian particle filters
and other sequential Monte Carlo methods. Given an array of `n`
non-negative weights, draws `n` indices into the array, each chosen
iid (so with replacement) with probability proportional to its
weight, in O(n) time. Implements the algorithm of Massey
(ICASSP 2008).

## Quick start

```rust
use ltsis::resample_indices;
use rand::SeedableRng;

let mut rng = rand::rngs::StdRng::seed_from_u64(42);

// Some weighted population. Weights need not be normalized.
let weights = vec![1.0_f32, 3.0, 2.0, 4.0];

// Resample 1000 indices from this distribution.
let mut out = vec![0_u32; 1000];
resample_indices(&mut rng, &weights, &mut out);

// `out[i]` ∈ {0, 1, 2, 3}; index 3 (weight 4.0) appears about 4×
// as often as index 0 (weight 1.0). Output is in ascending order.
//
// To use an output index for slice indexing, cast to usize:
//   let particle = particles[out[i] as usize];
```

## Background: Sequential Importance Resampling

In a Bayesian particle filter you carry a population of `n` candidate
states ("particles"), each with a weight that reflects how plausible
that state is given the data observed so far. As more data arrives,
the weights skew: most of the population ends up with negligible
weight, while a handful of particles dominate. **Resampling** refreshes
the population by drawing `n` new particles — each a copy of one of
the old ones, **with replacement** — where each old particle's chance
of being copied is proportional to its weight. Because draws are with
replacement, a high-weight particle typically appears multiple times
in the output; a low-weight one may not appear at all.

The fundamental operation: turn an array of weights into a multiset
of `n` indices, each chosen iid with probability proportional to
weight. This is exactly a multinomial draw on the weight distribution
— hence the standard name **multinomial resampling**. (Other
resampling schemes used in particle filters — systematic, residual,
stratified — produce a different joint distribution on the output
multiset; this crate doesn't implement those. If you want one of the
others, this isn't the crate for you.)

The naive way takes O(n log n) time (cumulative sum + binary search
per output) or O(n²) (scan per output). This crate runs in O(m + n) —
where `m = weights.len()` and `n = out.len()` — using a trick due to
Massey (2008): generate `n` sorted uniforms in `[0, 1)` in one O(n)
pass, then merge them against the cumulative weight array in another
O(m + n) pass.

## API

Two resamplers, identical signatures — pick whichever fits your
performance budget. Neither needs caller-supplied scratch.

- **`resample_indices(rng, weights, out)`** — streaming. One `powf`
  call per output index.
- **`resample_indices_buffered(rng, weights, out)`** — buffered.
  Generates sorted uniforms via Gamma ratios (Exp(1) draws) rather
  than `powf`. Typically ~1.28× faster on x86; more on hardware
  with a slow `powf`. Internally repurposes the `out` slice as
  scratch.

Plus two lower-level primitives:

- **`SortedUniforms::new(rng, n)`** — iterator yielding `n`
  Uniform(0, 1) variates in ascending order in O(n) time.
- **`first_uniform(rng, k)`** — sample `min(U₁, ..., Uₖ)` for
  k iid Uniform(0, 1) (equivalently, `Beta(1, k)`).

Indices are written as `u32` for platform-independent layout — not
`usize` — so callers cast at the index site
(`particles[out[i] as usize]`). Weight arrays must therefore have
length ≤ `u32::MAX`; debug builds assert this.

## Features

The crate has one feature axis: math source (`std` / `libm`, exactly
one).

- **`std`** (default): use std's inherent `f32::powf`.
- **`libm`**: use the [libm] crate via [num-traits] for `powf`.
  Suitable for bare-metal `no_std` targets.

All public APIs are `f32`. The library performs no allocation —
caller-supplied slices throughout — and is suitable for use on
Cortex-M4F and other single-precision FPU targets.

```toml
# Default (std):
[dependencies]
ltsis = "0.1"

# no_std:
[dependencies]
ltsis = { version = "0.1", default-features = false, features = ["libm"] }
```

[libm]: https://crates.io/crates/libm
[num-traits]: https://crates.io/crates/num-traits

## Performance

On modern x86 with a tuned libm: `first_uniform` runs at ~9 ns/call,
`resample_indices` at ~14.7 ns/step, `resample_indices_buffered` at
~11.4 ns/step (`black_box`-fenced microbench, scalar per-call
cost). On Cortex-M4F the buffered variant is expected to win by a
larger margin than on x86 because scalar `powf` is much more
expensive than the Exp(1) Ziggurat there. See `INTERNALS.md` for
methodology, full bench tables, and Cortex-M expectations.

## Testing & benchmarking

Integration tests:

```
cargo test --release
```

Ten statistical tests checking `first_uniform`, `SortedUniforms`,
and both resamplers against analytic CDFs, moment formulae, and an
independent oracle, using one-sample KS, two-sample KS, and chi-
squared statistics. Thresholds are calibrated so the aggregate
random-failure probability under correct code is **< 1e-9** (RNG
seeds fixed); methodology and threshold derivations are in
`INTERNALS.md` §5.4.

Microbenchmark:

```
cargo bench
```

Builds with `harness = false` (so it's a regular `fn main()`, not
the unstable `#[bench]` harness) and runs a hand-rolled
`black_box`-fenced timing loop for `first_uniform` and the full
resampling pipeline. See `INTERNALS.md` §5.5 for methodology and
§6 for typical numbers.

## Implementation notes & math proofs

For the algorithm specification, the floating-point boundary
argument that makes the merge memory-safe at f32 precision, the
Kahan-summation rationale, the `f32::to_bits` round-trip used in
the buffered variant, the formal correctness proofs (Lemmas 1–3,
Theorems 1–2), and a discussion of an open Padé-rational-squeeze
research direction, see [`INTERNALS.md`](INTERNALS.md).

## Citation

If you use this crate in academic work:

> Massey, B. (2008). Fast perfect weighted resampling.
> *Proceedings of IEEE ICASSP 2008*.

(plus this crate's repo URL).

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the
Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
