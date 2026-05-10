# `bmpf2` — Sequential Importance Resampling primitives

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
use bmpf2::resample_indices;
use rand::SeedableRng;

let mut rng = rand::rngs::StdRng::seed_from_u64(42);

// Some weighted population. Weights need not be normalized.
let weights = vec![1.0_f32, 3.0, 2.0, 4.0];

// Resample 1000 indices from this distribution.
let mut out = vec![0_u32; 1000];
resample_indices(&mut rng, &weights, &mut out);

// `out[i]` ∈ {0, 1, 2, 3}; index 3 (weight 4.0) appears about 4× as
// often as index 0 (weight 1.0). Output is in ascending order.
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
per output) or O(n²) (scan per output). This crate runs in O(m + n)
— where `m = weights.len()` and `n = out.len()` — using a trick due
to Massey (2008): generate `n` sorted uniforms in `[0, 1)` in one
O(n) pass, then merge them against the cumulative weight array in
another O(m + n) pass. The output is statistically equivalent to `n`
iid multinomial draws from the weight distribution.

Two resamplers are exposed, with **identical signatures** — pick
whichever fits your performance budget:

- `resample_indices` — **streaming**. One `powf` call per output
  index.
- `resample_indices_buffered` — **buffered**. Generates sorted
  uniforms via Gamma ratios (Exp(1) draws) rather than `powf`.
  Typically ~1.3× faster on x86; more on hardware with a slow
  `powf`. Internally repurposes the `out` slice as scratch (each
  `u32` slot temporarily holds an Exp draw's f32 bit pattern via
  `f32::to_bits`), then overwrites with the index in a second pass.

Neither resampler needs caller-supplied scratch.

Indices are written as `u32` for platform-independent layout — not
`usize` — so callers will need a `as usize` cast at the index site
(`particles[out[i] as usize]`). Weights arrays must therefore have
length ≤ `u32::MAX`; debug builds assert this.

The order-statistic iterator behind `resample_indices` is also
exposed as `SortedUniforms`, useful in its own right (e.g. for
inverse-CDF sampling against any continuous distribution where you
want sorted output).

If you want resampling and don't care how it works, the *Quick start*
above is all you need. The rest of this README covers the lower-level
building blocks, the math behind why it's correct, and the
floating-point arguments that keep it that way.

## Features

The crate has one feature axis: math source (`std` / `libm`, exactly one).

- `std` (default): use std's inherent `f32::powf`.
- `libm`: use the [libm] crate via [num-traits] for `powf`. Suitable for
  bare-metal `no_std` targets.

All public APIs are `f32`. The library performs no allocation —
caller-supplied slices throughout — and is suitable for use on Cortex-M4F
and other single-precision FPU targets. See the *Numerical robustness*
section below for the precision discussion.

```toml
# Default (std):
[dependencies]
bmpf2 = "0.1"

# no_std:
[dependencies]
bmpf2 = { version = "0.1", default-features = false, features = ["libm"] }
```

[libm]: https://crates.io/crates/libm
[num-traits]: https://crates.io/crates/num-traits

## Performance

**Watch out for autovectorization in microbenchmarks.** A naive benchmark
loop that accumulates samples lets LLVM emit a vectorized `powf` over
runs of consecutive iterations — measuring batched throughput rather
than scalar per-call cost. On Cortex-M4F (and any other no-SIMD
target) batched throughput is unattainable, so the relevant number is
*scalar* per-call cost. The shipped microbench wraps each call in
`std::hint::black_box` to defeat the cross-iteration vectorization
and measure scalar cost.

With `black_box` fences on a modern x86, `first_uniform` runs at
roughly 8.7–9.2 ns/sample across the full `k` range — essentially the
cost of one scalar `f32::powf` call (on this host's libm). On
Cortex-M4F, scalar `powf` is typically 100–150 cycles, so expect a
few times that on the same metric.

## Future work: rational-function squeeze

The library currently uses one `powf` per `first_uniform` call. An
earlier exploration around a polynomial "squeeze" to skip the
transcendental on the fast path was buggy (it ignored the M_k
normalization in the rejection-sampler formulation that was being
explored at the time, and produced biased samples — empirical mean for
k=2 came out 0.675 vs. theoretical 0.667). The original ICASSP 2008
paper's Ziggurat optimization had a similar normalization issue
(envelope geometry mismatched across n). Neither is retained here.

A more promising and not-yet-implemented direction is a **Padé
[m/m] rational squeeze** in a rejection scheme. The Padé[1/1] form of
`(1 − u)^(1/k)` is strikingly clean:

```
(1 − u)^(1/k)  ≈  (2k − (k+1)u) / (2k − (k−1)u)
```

so the spacing approximates `u / (k − (k−1)u/2)` (one mult, one
subtract, one divide; no transcendentals). Used directly as a sampler
this is biased — error grows large near `u → 1`, which is exactly the
rare-but-real "large spacing" tail — and the bias compounds in
`SortedUniforms` over `n = 10⁶` samples. Used as a one-sided **squeeze**
in a rejection sampler with `powf` on the slow path, the cost
amortizes: most attempts decide via the cheap rational test, and the
rare `powf` evaluation is statistically equivalent to no transcendental
at all for moderate-to-large `k`.

On host x86, this is unlikely to beat scalar `powf` (a well-tuned
x86 libm is hard to beat at ~9 ns/call). On Cortex-M4F, where scalar
`powf` is ~100–150 cycles, an all-multiplies-and-adds fast path
could plausibly win 30–50%. Whether that's worth the additional
code complexity (correctness proof for the one-sidedness of the
bound, careful slow-path correctness) depends on the deployment
target. Filed as a future direction; happy to revisit if a Cortex-M
deployment ever materializes that needs it.

## Mathematical correctness

This section establishes that `resample_indices` produces a sample with
exactly the same joint distribution as drawing `n` independent multinomial
variates from the weight distribution and then sorting them. The argument
is in three pieces: (1) the sorted-uniforms generator produces the right
joint distribution; (2) the merge converts sorted uniforms into a sorted
multinomial sample; (3) the floating-point boundary argument is sound.

**Background and prior work.** The order-statistic recurrence used by
`SortedUniforms` is classical. The earliest formulation we are aware of
in the simulation literature is Bentley & Saxe (1980), who give a
sequential O(n) algorithm for generating sorted uniforms by drawing
each one as a function of the previous one and a fresh uniform deviate.
Devroye (1986, Chapter V, especially §V.3.1) gives a textbook treatment
including the spacings approach used here. Within particle-filter
practice, Carpenter, Clifford & Fearnhead (1999) note the spacings
construction and use it for sampling-importance resampling. The
resampling algorithm in this crate combines the Bentley–Saxe sorted
generator with a standard merge against cumulative weights; the same-pass
combination was the contribution of Massey (2008). The novelty of the
present work over the 2008 paper is correctness of the variate
implementation (the original paper's Ziggurat was buggy; see "Future
work: rational-function squeeze" above) and a careful floating-point
boundary argument for the merge.

### Lemma 1 (memorylessness of uniform order statistics)

Let `U₁, ..., Uₙ` be i.i.d. Uniform(0, 1) and let
`U₍₁₎ ≤ U₍₂₎ ≤ ... ≤ U₍ₙ₎` be their order statistics. For any
1 ≤ i ≤ n − 1, conditional on `U₍ᵢ₎ = u`, the remaining order statistics
`U₍ᵢ₊₁₎, ..., U₍ₙ₎` are jointly distributed as the order statistics of
n − i i.i.d. Uniform(u, 1) variates.

*Proof.* This is a standard property of order statistics from a
continuous distribution; see e.g. Devroye (1986), §V.3, or David & Nagaraja,
*Order Statistics* (3rd ed., 2003), §2.4. The key fact is that conditional
on `U₍ᵢ₎ = u`, the values `Uⱼ` exceeding `u` are i.i.d. Uniform(u, 1)
by the memoryless construction of order statistics.       ∎

### Lemma 2 (minimum of k uniforms)

If `V₁, ..., Vₖ` are i.i.d. Uniform(0, 1) then `min(V₁, ..., Vₖ)` has CDF
`F(v) = 1 − (1 − v)ᵏ` on [0, 1] — i.e., `min(V₁, ..., Vₖ) ~ Beta(1, k)`.

*Proof.* `Pr[min Vᵢ > v] = Pr[V₁ > v] · ... · Pr[Vₖ > v] = (1 − v)ᵏ`.       ∎

### Theorem 1 (correctness of `SortedUniforms`)

The iterator `SortedUniforms::new(rng, n)` yields a sequence of values
distributed as the order statistics of n i.i.d. Uniform(0, 1) variates.

*Proof.* By induction on the iteration index `i ∈ {1, ..., n}`. Write
`lastᵢ` for the value of the internal `last` variable just after the i-th
yield, with `last₀ = 0`.

*Base case (i = 1).* On the first call, `remaining = n` and `last = 0`.
The iterator computes `spacing = first_uniform(rng, n)`. By construction
of `first_uniform`, this `spacing` is distributed as `Beta(1, n)`, which
by Lemma 2 is the distribution of the minimum of n i.i.d. Uniform(0, 1)
variates — i.e., the distribution of `U₍₁₎`. The yielded value is
`last + (1 − last) · spacing = spacing`, and `last₁ = spacing`. So
`last₁ ~ U₍₁₎`. ✓

*Inductive step.* Assume that after i yields, `(last₁, ..., lastᵢ)` has
the same joint distribution as `(U₍₁₎, ..., U₍ᵢ₎)`, the first i order
statistics of n i.i.d. Uniform(0, 1). The iterator now has
`remaining = n − i` and proceeds:

```
spacing = first_uniform(rng, n − i)
yield   = lastᵢ + (1 − lastᵢ) · spacing
lastᵢ₊₁ = yield
```

By Lemma 1, conditional on `lastᵢ = u`, the remaining order statistics
`U₍ᵢ₊₁₎, ..., U₍ₙ₎` are distributed as the order statistics of n − i
i.i.d. Uniform(u, 1) variates. In particular, `U₍ᵢ₊₁₎` is the minimum of
n − i such variates, so

```
U₍ᵢ₊₁₎ - u  ~  (1 − u) · min(W₁, ..., Wₙ₋ᵢ)
```

where `W₁, ..., Wₙ₋ᵢ` are i.i.d. Uniform(0, 1). By Lemma 2,
`min(W₁, ..., Wₙ₋ᵢ) ~ Beta(1, n − i)`, which is exactly the
distribution of `first_uniform(rng, n − i)` — i.e. `spacing`.
Therefore `lastᵢ₊₁ = lastᵢ + (1 − lastᵢ) · spacing` has the same
conditional distribution given `lastᵢ` as `U₍ᵢ₊₁₎` has given `U₍ᵢ₎`,
and the inductive hypothesis extends to step i + 1.       ∎

### Theorem 2 (correctness of `resample_indices`)

Let `w₁, ..., wₘ ≥ 0` with `T = Σⱼ wⱼ > 0`. Define i.i.d. multinomial
draws `K₁, ..., Kₙ` with `Pr[Kₐ = j] = wⱼ / T` for j ∈ {1, ..., m}. Then
the output sequence `J₁ ≤ J₂ ≤ ... ≤ Jₙ` produced by
`resample_indices(rng, weights, out)` (with `out.len() = n`) has the
same joint distribution as the sorted multinomial sample
`(K₍₁₎, ..., K₍ₙ₎)` (with 1-based indices in this proof; the code is
0-indexed).

*Proof.* Define `Wⱼ = w₁ + ... + wⱼ` (cumulative weights, `W₀ = 0`,
`Wₘ = T`) and `F(j) = Wⱼ / T` (the multinomial CDF). The inverse-CDF
construction of multinomial sampling is: for each a ∈ {1, ..., n},
draw `Uₐ ~ Uniform(0, 1)` and set

```
Kₐ = min { j ∈ {1, ..., m} : F(j) > Uₐ }
   = min { j : T · Uₐ < Wⱼ }.                                         (*)
```

This is correct because `F(j − 1) ≤ Uₐ < F(j)` happens with probability
`F(j) − F(j − 1) = wⱼ / T`, the desired multinomial probability.

The monotonicity of `j ↦ F(j)` makes the map `Uₐ ↦ Kₐ` monotone
non-decreasing. Therefore sorting the `Uₐ` ascending and mapping each
through (∗) yields the sorted multinomial sample:

```
(K₍₁₎, ..., K₍ₙ₎) = (φ(U₍₁₎), ..., φ(U₍ₙ₎))                          (**)
```

where `φ(u) = min { j : T·u < Wⱼ }`. (When there is a tie in the `Uₐ`,
`φ` is constant on the tie, so the equality holds.)

Now examine what `resample_indices` does. By Theorem 1, the iterator
yields `(U₍₁₎, ..., U₍ₙ₎)` distributed as the order statistics of n
i.i.d. Uniform(0, 1). For each yielded `U₍ᵢ₎`, the merge loop sets
`target = T · U₍ᵢ₎` and advances j until `target ≤ cumulative`, where
`cumulative` is the running prefix sum `Wⱼ`. The merge therefore computes

```
Jᵢ = min { j : Wⱼ ≥ T·U₍ᵢ₎ }.                                       (***)
```

Predicates (∗) and (∗∗∗) differ only on the event `T·U = Wⱼ` for some
j — i.e., on a finite union of points in [0, 1] — which has probability
zero under the continuous uniform distribution. So `Jᵢ = φ(U₍ᵢ₎)` almost
surely, and the joint distributions match (∗∗).       ∎

### Lemma 3 (floating-point boundary)

Suppose `weights[i]` are finite and nonnegative with positive sum. Then
`resample_indices` cannot index past `weights.len() − 1` regardless of
the values yielded by `SortedUniforms`, provided those values lie in
[0, 1].

*Proof.* `total` is computed by Kahan compensated summation walking
`weights` in index order, starting from compensated state
`(sum, c) = (0, 0)`:

```
(total, total_c) = (0, 0)
for w in weights:
    (total, total_c) = kahan_add(total, total_c, w)
```

When `j` reaches `m − 1` in the merge, `(cumulative, cumulative_c)`
has been built by initializing `(cumulative, cumulative_c) = (weights[0], 0)`
(equivalent to one Kahan step from `(0, 0)`, since the first
compensator update yields zero) and then executing
`kahan_add(cumulative, cumulative_c, weights[k])` for k = 1, 2, ...,
m − 1 in order — visiting weights in the same order with the same
accumulator algorithm. IEEE 754 binary floating-point addition is
deterministic and a function only of its operands and rounding mode
(round-to-nearest in Rust by default), so Kahan summation, built
purely from such adds and subtracts, is also deterministic. The two
sequences therefore yield bit-identical results: `cumulative == total`
exactly when `j = m − 1`.

Each value `u` yielded by `SortedUniforms` satisfies `u ≤ 1` (the
recurrence preserves this; see edge-case note below). Furthermore,
since `total` is an `f32`-representable value and `u ∈ [0, 1]`, the
exact product `total · u` is bounded above by `total`; in
round-to-nearest f32 this rounds to a representable value `≤ total`
(because `total` itself is representable, every value `< total`
rounds to a representable value `≤ total`). Therefore
`target ≤ total = cumulative` whenever `j = m − 1`, and the strict
inequality `target > cumulative` is false: the while loop exits with
`j = m − 1` rather than incrementing further.       ∎

**Edge case.** `first_uniform` returns values strictly in `[0, 1)` by
construction. It computes `1 − (1 − u)^(1/k)` for
`u ~ rng.gen::<f32>()` (i.e. `u ∈ [0, 1 − 2⁻²⁴]`). Since `1 − u` lands
in `[2⁻²⁴, 1]` exactly in f32, `(1 − u)^(1/k) ∈ [2⁻²⁴ᐟᵏ, 1]`, and the
output is in `[0, 1 − 2⁻²⁴ᐟᵏ] ⊂ [0, 1)`. Each of the 2²⁴ input bins
maps to a distinct output, all in range — no redraw, no saturation,
no special case. There is a benign `f32` rounding artifact:
`(1 − u)^(1/k)` can round to exactly `1` for very large `k` and `u`
near `0`, making the output `0` and `spacing = 0`; the recurrence
then yields the prior `last` again — a vanishing statistical artifact
(the f32 quantization of "consecutive order statistics rounded to the
same value") and not a Lemma 3 violation since `last ≤ 1` is
preserved.

So `last` never reaches 1 before the final yield, and the distribution
of the yielded variates matches Theorem 1 to within `f32` quantization.

## Numerical robustness

The library is `f32` throughout, with `ε = 2⁻²⁴ ≈ 6·10⁻⁸`. Beyond the
boundary argument, the precision concern is the prefix sums: a naive
running sum over `n` non-negative terms accrues relative error of
order `O(n · ε)`, which becomes unusable around `n ≈ 10⁵`. The
library therefore uses **Kahan compensated summation** for every
multi-term accumulator on the data path:

- `total = Σ weights` and the merge's incremental `cumulative_w` in
  both `resample_indices` and `resample_indices_buffered`.
- `G = Σ E_i` and the merge's `cumulative_e = S_i = E_1 + ... + E_i`
  in `resample_indices_buffered`.

This reduces the bound on each accumulator's relative error from
`O(n · ε)` to `O(ε)` — effectively constant in `n`.

Producing exact bit-identical `total` and `cumulative` at the right
endpoint depends on the same accumulation algorithm and traversal
order being used in both places. This is guaranteed in the current
implementation by both passes Kahan-summing `weights` in increasing
index order from `(sum, c) = (0, 0)`; changing either to a parallel
reduce, a different traversal, or a non-Kahan accumulator would
require revisiting Lemma 3.

`resample_indices_buffered` additionally applies a `target.min(total)`
clip in the merge to handle the case where `u_n = S_n / G` rounds up
to exactly `1.0` in `f32` (it can, with probability `~3%` at
`n = 10⁶`, when `E_{n+1}/G` falls below `~2⁻²⁵`). This is a separate
issue from the Lemma 3 boundary argument and not addressed by Kahan.

## Citations

- Bentley, J. L. & Saxe, J. B. (1980). Generating sorted lists of random
  numbers. *ACM Transactions on Mathematical Software*, 6(3), 359–364.
  Original linear-time sequential generator for sorted uniforms.
- Carpenter, J., Clifford, P. & Fearnhead, P. (1999). An improved
  particle filter for non-linear problems. *IEE Proceedings — Radar,
  Sonar and Navigation*, 146(1), 2–7. Spacings construction in the SIR
  context.
- David, H. A. & Nagaraja, H. N. (2003). *Order Statistics*, 3rd ed.
  Wiley. Comprehensive treatment of joint distribution of order
  statistics.
- Devroye, L. (1986). *Non-Uniform Random Variate Generation.* Springer.
  Chapter V (Uniform and Exponential Spacings); §V.3.1 specifically
  treats the generation of uniform [0, 1] order statistics. Available
  free at https://luc.devroye.org/rnbookindex.html.
- Massey, B. (2008). Fast perfect weighted resampling. *Proceedings of
  IEEE ICASSP 2008*. The merge-with-sorted-uniforms construction this
  crate implements; see also the `icassp-ltrs.pdf` companion document
  in this repository's design history.

## Resampling performance

Full-pipeline benchmark (`m = n`, weights = `1..=m`, `f32`,
host-specific, all calls `black_box`-fenced):

```
    m = n     streaming                buffered              streaming
              ns/call    ns/step       ns/call    ns/step    /buffered
      100         2938     14.69          2294     11.47        1.28x
     1000        29668     14.83         23149     11.57        1.28x
    10000       293323     14.67        230499     11.52        1.27x
   100000      2918029     14.59       2275946     11.38        1.28x
  1000000     29067435     14.53      22662423     11.33        1.28x
```

Linear scaling is clean across five orders of magnitude. Per-step
cost is dominated by the `Beta(1, k)` draw (streaming) or the
`Exp(1)` draw plus merge (buffered); cache effects don't materially
degrade large-`m` performance because the weight array is touched in
a single forward sweep.

The ~1.28× ratio on this host reflects the per-call cost difference
between scalar `powf` and an `Exp(1)` Ziggurat draw on a well-tuned
x86 libm. On Cortex-M4F, where scalar `powf` is much slower
(~100–150 cycles vs ~30–60 for `Exp(1)` Ziggurat), the buffered
variant is expected to win by a larger margin.

## Files

- `src/lib.rs` — top-level docs, re-exports, feature compile-errors.
- `src/first_uniform.rs` — the [`first_uniform`] sampler.
- `src/sorted_uniforms.rs` — `SortedUniforms` iterator.
- `src/resample.rs` — `resample_indices` and
  `resample_indices_buffered`, plus the private Kahan-add helper.
- `src/bin/tests.rs` — statistical correctness driver and microbench
  (run via `cargo run --bin tests`; intentionally not on the
  `cargo test` path because the tests are tolerance-based, not crisp
  invariants).
- `Cargo.toml` — package manifest with feature gates.

## Building

```
# Default (std):
cargo run --release --bin tests

# no_std smoke test (lib only — bin is gated to std):
cargo build --release --no-default-features --features libm
```

For genuine `no_std` verification, build against an embedded target,
e.g.: `cargo build --release --no-default-features --features libm
--target thumbv7em-none-eabihf`.
