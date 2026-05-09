# `bmpf2` — Sequential Importance Resampling primitives

A Rust library for the per-step variates and merge needed by linear-time
perfect weighted resampling (Sequential Importance Resampling, SIR), used
in Bayesian particle filters and elsewhere. Implements the algorithm of
Massey (ICASSP 2008).

The core variate is the **minimum of `k` iid Uniform(0, 1) draws** —
equivalently `Beta(1, k)` in the standard parametrization — which is the
per-step "spacing" distribution in the order-statistic recurrence used to
generate `n` sorted uniforms in O(n) time. Exposed as
[`first_uniform`](https://docs.rs/bmpf2) (with named-backend variants
`first_uniform_pow` and `first_uniform_rejection`).

## Features

The crate has two orthogonal feature axes: backend (`pow` / `rejection`,
exactly one) and math source (`std` / `libm`, exactly one).

Backend:
- `pow` (default): direct `u.powf(1.0 / k)`.
- `rejection`: Exp(1)-proposal rejection sampling with log-space acceptance.

Math source:
- `std` (default): use std's inherent `f32::ln` / `f32::powf`.
- `libm`: use the [libm] crate via [num-traits] for `ln` / `powf`.
  Suitable for bare-metal `no_std` targets.

`first_uniform` dispatches to whichever backend is enabled (preferring
`pow` if both are enabled). The two underlying functions
`first_uniform_pow` and `first_uniform_rejection` are always available
regardless of features, so tests and benchmarks can exercise both.

All public APIs are `f32`. The library performs no allocation —
caller-supplied slices throughout — and is suitable for use on Cortex-M4F
and other single-precision FPU targets. See the *Numerical robustness*
section below for the precision discussion.

```toml
# Default (std + pow):
[dependencies]
bmpf2 = "0.1"

# no_std with the rejection backend:
[dependencies]
bmpf2 = { version = "0.1", default-features = false, features = ["rejection", "libm"] }
```

[libm]: https://crates.io/crates/libm
[num-traits]: https://crates.io/crates/num-traits

## Algorithm: Exp(1)-proposal rejection

For k ≥ 2, the rejection sampler uses an Exp(1) proposal:

```
draw Y ~ Exp(1)
if Y ≥ k: reject
draw V ~ Uniform(0, 1)
accept iff log V < (k-1)·log(1 - Y/k) + Y - log M_k
return X = Y / k
```

where `log M_k = (k-1) · log(1 - 1/k) + 1`. For k = 1, return U directly
(`Beta(1, 1)` is just Uniform).

## Proof of correctness for the rejection sampler

**Setup.** The target density is `f_k(x) = k · (1 - x)^(k-1)` on
`[0, 1]` (i.e. `Beta(1, k)`). Apply the change of variables `Y = k · X`,
so `X = Y/k` and `dY = k · dX`. The density of Y is

```
f_Y(y) = f_k(y/k) · (1/k) = (1 - y/k)^(k-1)        for y ∈ [0, k].
```

**Proposal.** We propose Y' ~ Exp(1), with density `g(y) = e^(-y)` on
`[0, ∞)`. To use rejection sampling, we need a constant `M_k` such that
`f_Y(y) ≤ M_k · g(y)` for all `y ∈ [0, k]`, i.e.

```
(1 - y/k)^(k-1) · e^y ≤ M_k                            for all y ∈ [0, k].
```

**Computing M_k.** Let `h_k(y) = (1 - y/k)^(k-1) · e^y`. Take logs:
`log h_k(y) = (k-1) · log(1 - y/k) + y`. Differentiate:
`d/dy log h_k(y) = -(k-1)/(k - y) + 1`. Setting to zero gives `y = 1`.
Second derivative is `-(k-1)/(k-y)^2 < 0`, so `y = 1` is a maximum.
Therefore

```
M_k = h_k(1) = (1 - 1/k)^(k-1) · e          (*)
```

This M_k is finite for all k ≥ 1 (and approaches 1 as k → ∞). For k = 2,
`M_2 = (1/2) · e ≈ 1.359`. For k = 10, `M_10 ≈ 1.105`. For large k,
`M_k → e^(1/k) → 1`.

**Acceptance probability.** Standard rejection sampling: given proposal
Y, accept with probability
`A_k(Y) = f_Y(Y) / [M_k · g(Y)] = (1 - Y/k)^(k-1) · e^Y / M_k`. By
construction (*), `A_k(Y) ≤ 1` for all Y ∈ [0, k]. We perform this
acceptance test in log space:

```
log A_k(Y) = (k-1) · log(1 - Y/k) + Y - log M_k
```

and accept iff `log V < log A_k(Y)` for V ~ Uniform(0, 1) (equivalently,
iff a fresh uniform is below the true acceptance probability).

**Outside the support.** When Y ≥ k, we reject outright and redraw. This
is required because the target Y is supported only on [0, k]. The
probability of this rejection is `Pr[Y ≥ k] = e^(-k)`, which is tiny for
any practical k.

**Output.** Conditional on acceptance, Y is distributed as `f_Y`, so
`X = Y/k` is distributed as `f_k = Beta(1, k)`. ∎

## Performance

**Watch out for autovectorization in microbenchmarks.** A naive benchmark
loop that accumulates samples will, on x86 with a SIMD libm, let LLVM emit
vectorized `pow` over runs of consecutive iterations — inflating apparent
`pow` throughput far above what a Cortex-M can achieve. The shipped
microbench wraps each call in `std::hint::black_box` to defeat this.

With `black_box` fences on a modern x86 (f32 path):

```
  k     pow (ns/sample)    rejection (ns/sample)    rej/pow
  2           9.30                  28.66            3.08x
  5           9.34                  24.17            2.59x
  10          9.29                  21.96            2.36x
  50          9.33                  20.63            2.21x
  200         9.29                  20.14            2.17x
  1000        9.24                  20.00            2.16x
```

Without `black_box`, SIMD vectorization on the same host reports `pow`
several times faster than this. That throughput is real for
vectorizable workloads but is not representative of scalar per-call
cost on hardware without SIMD.

The `pow` path wins on this host. On a Cortex-M4F with a typical
embedded libm, where scalar `powf` may take 100+ cycles versus a
`logf` of 30–50 cycles plus an Exp(1) draw, the rejection path may
win — particularly for moderate to large k where the M_k overhead is
small. **Benchmark on your actual target before choosing.**

## Background: the squeeze that wasn't

Earlier design discussions explored a quadratic squeeze
`squeeze(Y) = Y² · (k-1) / (2k)` to skip the `log` call on the fast path.
That construction was wrong: it ignored the M_k normalization and so
produced biased samples for small k. Empirical mean for k=2 was 0.675 vs.
theoretical 0.667, clearly detected by the KS test. The shipped library
drops the squeeze and always evaluates the log-acceptance test; this
costs one `log` per attempt. The original ICASSP 2008 paper's Ziggurat
optimization had a similar M-related issue (envelope geometry mismatched
across n) and is not retained here.

## Linear-time perfect weighted resampling

The crate provides the resampling algorithm that motivated the
`first_uniform` sampler in the first place.

```rust
use bmpf2::resample_indices;

let weights = vec![1.0, 3.0, 2.0, 4.0];
let mut out = vec![0usize; 1000];
resample_indices(&mut rng, &weights, &mut out);
// out now contains 1000 indices into `weights`, drawn with probability
// proportional to weight, sorted ascending.
```

`resample_indices` runs in O(`weights.len()` + `out.len()`) time. It is
streaming — one `first_uniform` (and hence one `pow` or one rejection
attempt) per output index, no scratch buffer. A buffered variant trades
one extra `n`-element scratch buffer for faster per-element cost (no
`first_uniform` per element):

```rust
use bmpf2::resample_indices_buffered;

let weights = vec![1.0, 3.0, 2.0, 4.0];
let mut out = vec![0usize; 1000];
let mut scratch = vec![0.0_f32; 1000];   // length must equal out.len()
resample_indices_buffered(&mut rng, &weights, &mut out, &mut scratch);
```

On the host x86 with the `pow` backend, `resample_indices_buffered`
runs at ~12 ns/step against ~15 ns/step for `resample_indices` (~1.30×
faster). On hardware with a slower `powf` (e.g. Cortex-M4F), the gap
is expected to widen.

The streaming sorted-uniforms generator is also exposed:

```rust
use bmpf2::SortedUniforms;
for u in SortedUniforms::new(&mut rng, 100) {
    // u_1 ≤ u_2 ≤ ... ≤ u_100, all in [0, 1]
}
```

The output indices from both resamplers come out in ascending order,
which is what most downstream code (e.g. gathering particles into a
new buffer) wants anyway. If you need them shuffled, do that as a
post-pass.

### Mathematical correctness

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
implementation (the original paper's Ziggurat was buggy; see the
"squeeze that wasn't" section above) and a careful floating-point
boundary argument for the merge.

#### Lemma 1 (memorylessness of uniform order statistics)

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

#### Lemma 2 (minimum of k uniforms)

If `V₁, ..., Vₖ` are i.i.d. Uniform(0, 1) then `min(V₁, ..., Vₖ)` has CDF
`F(v) = 1 − (1 − v)ᵏ` on [0, 1] — i.e., `min(V₁, ..., Vₖ) ~ Beta(1, k)`.

*Proof.* `Pr[min Vᵢ > v] = Pr[V₁ > v] · ... · Pr[Vₖ > v] = (1 − v)ᵏ`.       ∎

#### Theorem 1 (correctness of `SortedUniforms`)

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

#### Theorem 2 (correctness of `resample_indices`)

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

#### Lemma 3 (floating-point boundary)

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

**Edge case.** Both backends return values strictly in `[0, 1)` by
construction:

- `first_uniform_pow` computes `1 − (1 − u)^(1/k)` for
  `u ~ rng.gen::<f32>()` (i.e. `u ∈ [0, 1 − 2⁻²⁴]`). Since `1 − u`
  lands in `[2⁻²⁴, 1]` exactly in f32, `(1 − u)^(1/k) ∈ [2⁻²⁴ᐟᵏ, 1]`,
  and the output is in `[0, 1 − 2⁻²⁴ᐟᵏ] ⊂ [0, 1)`. Each of the 2²⁴
  input bins maps to a distinct output, all in range — no redraw,
  no saturation, no special case. There is a benign `f32` rounding
  artifact: `(1 − u)^(1/k)` can round to exactly `1` for very large
  `k` and `u` near `0`, making the output `0` and `spacing = 0`;
  the recurrence then yields the prior `last` again, which is a
  vanishing statistical artifact (the f32 quantization of
  "consecutive order statistics rounded to the same value") and not
  a Lemma 3 violation since `last ≤ 1` is preserved.
- The `rejection` backend returns `y/k` with `y > 0` strictly, so its
  output is in `(0, 1)` exactly.

In neither backend does `last` reach 1 before the final yield, so
the distribution of the yielded variates matches Theorem 1 to within
`f32` quantization.

### Numerical robustness

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

### Citations

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

### Resampling performance

Full-pipeline benchmark (m = n, weights = 1..=m, `f32`, host-specific):

```
    m = n     C ns/call   C ns/step    B ns/call   B ns/step    C/B
      100         3174       15.87         2425       12.12     1.31x
     1000        31968       15.98        24453       12.23     1.31x
    10000       314165       15.71       242873       12.14     1.29x
   100000      3108815       15.54      2399701       12.00     1.30x
  1000000     31075298       15.54     23910144       11.96     1.30x
```

C = `resample_indices` (streaming Method C, no scratch). B =
`resample_indices_buffered` (Method B, requires `n`-element scratch).
Linear scaling is clean across five orders of magnitude. The per-step
cost is dominated by the Beta(k, 1) draw (Method C) or the Exp(1) draw
plus the merge (Method B); cache effects don't materially degrade
large-m performance because the weight array is touched in a single
forward sweep.

The C/B ratio of ~1.3× on this host (modern x86 with SIMD libm)
narrows because vectorized `pow` is fast here. On Cortex-M4F with no
SIMD and a slower scalar `powf`, Method B is expected to win by a
larger margin.

## Files

- `src/lib.rs` — top-level docs, re-exports, feature compile-errors.
- `src/first_uniform.rs` — `first_uniform` dispatcher and the two
  named-backend variants `first_uniform_pow` / `first_uniform_rejection`.
- `src/sorted_uniforms.rs` — `SortedUniforms` iterator.
- `src/resample.rs` — `resample_indices` (Method C) and
  `resample_indices_buffered` (Method B), plus the private Kahan-add
  helper.
- `src/bin/tests.rs` — statistical correctness driver and microbench
  (run via `cargo run --bin tests`; intentionally not on the
  `cargo test` path because the tests are tolerance-based, not crisp
  invariants).
- `Cargo.toml` — package manifest with feature gates.

## Building

```
# Default (std + pow):
cargo run --release --bin tests

# std + rejection:
cargo run --release --bin tests --no-default-features --features std,rejection

# no_std smoke tests (lib only — bin is gated to std):
cargo build --release --no-default-features --features libm,pow
cargo build --release --no-default-features --features libm,rejection
```

For genuine `no_std` verification, build against an embedded target,
e.g.: `cargo build --release --no-default-features --features libm,pow
--target thumbv7em-none-eabihf`.
