# `ltsis` — linear-time Sequential Importance Sampling primitives
Bart Massey and Claude 2026

[![Crates.io](https://img.shields.io/crates/v/ltsis.svg)](https://crates.io/crates/ltsis)
[![Documentation](https://docs.rs/ltsis/badge.svg)](https://docs.rs/ltsis)
[![CI](https://github.com/BartMassey/ltsis/actions/workflows/ci.yml/badge.svg)](https://github.com/BartMassey/ltsis/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/crates/l/ltsis.svg)](#license)

A Rust library for **Sequential Importance Sampling (SIS) with
replacement** — also known as **multinomial sampling** — the
weighted-sampling step at the heart of Bayesian particle filters
and other sequential Monte Carlo methods. Given an array of `n`
non-negative weights, draws `n` indices into the array, each chosen
iid (so with replacement) with probability proportional to its
weight, in $O(n)$ time. Implements the algorithm of Massey
(ICASSP 2008).

## Quick start

```rust
use ltsis::sample_indices;
use rand::SeedableRng;

let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

// Some weighted population. Weights need not be normalized.
let weights = vec![1.0_f32, 3.0, 2.0, 4.0];

// `sample_indices` returns an iterator yielding 1000 indices.
// Collect, fold into your own buffer, or consume in a loop.
let out: Vec<u32> = sample_indices(&mut rng, &weights, 1000).collect();

// Each yielded index ∈ {0, 1, 2, 3}; index 3 (weight 4.0) appears
// about 4× as often as index 0 (weight 1.0). Indices are yielded
// in ascending order.
//
// To use a yielded index for slice indexing, cast to usize:
//   let particle = particles[j as usize];
```

## Background: Sequential Importance Sampling

Sequential Importance Sampling (SIS) samples a distribution
by drawing `n` new samples from a distribution of `m` old
ones. The old samples have associated weights: SIS selects
each new sample independently from the old distribution,
with a sample having selection probability proportional to
the weight.

For example, in a Bayesian particle filter you carry a
population of `n` candidate states ("particles"), each with
a weight that reflects how plausible that state is given the
data observed so far. As more data arrives, the weights
skew: most of the population ends up with negligible weight,
while a handful of particles dominate. **Sampling**
refreshes the population by drawing `n` new particles — each
a copy of one of the old ones, **with replacement** — where
each old particle's chance of being copied is proportional
to its weight. Because draws are with replacement, a
high-weight particle typically appears multiple times in the
output; a low-weight one may not appear at all.

The fundamental operation: turn an array of weights into a multiset
of `n` indices, each chosen iid with probability proportional to
weight. This is exactly a multinomial draw on the weight distribution
— hence the standard name **multinomial sampling**. (Other
sampling schemes used in particle filters — systematic, residual,
stratified — produce a different joint distribution on the output
multiset; this crate doesn't implement those.)

When implemented naïvely, SIS takes $O(n m)$ time to produce
$n$ samples from $m$ weights: the textbook inverse-CDF
construction walks the weight array once per output,
accumulating a prefix sum until it crosses a uniform threshold
(an $O(m)$ operation per sample on average).

Precomputing the cumulative-weight array up front and binary-
searching it brings the per-sample cost down to $O(\log m)$,
for $O(m + n \log m)$ total. This crate runs in $O(m + n)$ —
using a trick due to Massey (2008): generate $n$ sorted
uniforms in $[0, 1)$ in one $O(n)$ pass, then merge them
against the cumulative weight array in another $O(m + n)$
pass.

## API

Two samplers are provided.

- **`sample_indices(rng, weights, n) -> impl Iterator<Item = u32>`**
  — streaming. Returns an iterator yielding `n` indices in
  ascending order. One `powf` call per yielded index. No
  allocation.

- **`sample_indices_buffered(rng, weights, &mut out)`** — buffered.
  Generates sorted uniforms via Gamma ratios (Exp(1) draws) rather
  than `powf`. Typically ~1.3× faster on x86; more on hardware
  with a slow `powf`. Takes an `&mut [u32]` rather than returning
  an iterator because it repurposes the buffer's slots as f32
  scratch.

These are built on two lower-level public primitives:

- **`first_uniform(rng, k)`** — sample $\min(U_1, \ldots, U_k)$
  for $U_k$ drawn iid from $\mathrm{Uniform}(0, 1)$. Constant time.

- **`SortedUniforms::new(rng, n)`** — iterator yielding $n$
  $\mathrm{Uniform}(0, 1)$ variates in ascending order in $O(n)$ time.
  Uses `first_uniform()` internally, so constant-time per
  iteration.

Output indices are `u32` rather than `usize` for platform
independence and for internal reasons; this is slightly
inconvenient for the API user, but avoids a host of issues.
Weight arrays must therefore have length ≤ `u32::MAX`.

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
[dependencies.ltsis]
version = "0.1"
default-features = false
features = ["libm"]
```

[libm]: https://crates.io/crates/libm
[num-traits]: https://crates.io/crates/num-traits

## Performance

On modern x86 with a tuned libm: `first_uniform` runs at ~10 ns/call,
`sample_indices` at ~14 ns/step, `sample_indices_buffered` at
~11 ns/step (`black_box`-fenced microbench, scalar per-call cost,
SmallRng/Xoshiro256++). On Cortex-M4F the buffered variant is
expected to win by a larger margin than on x86 because scalar
`powf` is much more expensive than the Exp(1) Ziggurat there. See
`INTERNALS.md` for methodology, full bench tables, and Cortex-M
expectations.

## Testing & benchmarking

### Integration tests

```
cargo test --release
```

Ten statistical tests checking `first_uniform`, `SortedUniforms`,
and both samplers against analytic CDFs, moment formulae, and an
independent oracle, using one-sample KS, two-sample KS, and chi-
squared statistics. Thresholds are calibrated so the aggregate
random-failure probability under correct code is **< 1e-9** (RNG
seeds fixed); methodology and threshold derivations are in
`INTERNALS.md` §5.4.

### Microbenchmark

```
cargo bench
```

Uses [Divan](https://docs.rs/divan) (per-iteration timing,
outlier rejection, tabular summary) for `first_uniform` per-call
cost and for the full sampling pipeline in both `black_box`-
fenced and unfenced modes. See `INTERNALS.md` §5.5 for
methodology and §6 for typical numbers.

## Technical details

See [`INTERNALS.md`](INTERNALS.md) for a detailed discussion
of design and implementation, including proofs and proof
citations.

If you use this crate in academic work, you might cite

> Massey, B. (2008). Fast perfect weighted resampling.
> *Proceedings of IEEE ICASSP 2008*.

plus this crate's repo URL.

## License

Dual-licensed under either of:

- "Apache License, Version 2.0" ([LICENSE-APACHE](LICENSE-APACHE)).
- "MIT license" ([LICENSE-MIT](LICENSE-MIT)).

at your option. Please see the license files in this
distribution for license terms.

### Contribution

Unless you explicitly state otherwise, any contribution
intentionally submitted for inclusion in the work by you, as
defined in the Apache-2.0 license, shall be dual licensed as
above, without any additional terms or conditions.

