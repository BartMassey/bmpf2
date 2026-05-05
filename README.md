# `beta_k_1` — sampling Beta(k, 1) for linear-time perfect resampling

A Rust library for drawing X ~ Beta(k, 1), the distribution of the maximum
of k i.i.d. Uniform(0, 1) variates. Equivalently, X is distributed as
`U^(1/k)` for U ~ Uniform(0, 1). This is the per-step variate distribution
needed by the linear-time perfect weighted resampling algorithm of Massey
(ICASSP 2008), where k decreases from n down to 1 over the course of a
resampling pass.

## Features

The crate exposes both implementations under feature gates:

- `pow` (default): direct `u.powf(1.0 / k)`.
- `rejection`: Exp(1)-proposal rejection sampling with log-space acceptance.

`beta_k_1` dispatches to whichever is enabled (preferring `pow` if both are
enabled, with `pow` being the most common case). The two underlying
functions `beta_k_1_pow` and `beta_k_1_rejection` are always available
regardless of features, so tests and benchmarks can exercise both.

```toml
[dependencies]
beta_k1 = { version = "0.1", default-features = false, features = ["rejection"] }
```

## Algorithm: Exp(1)-proposal rejection

For k ≥ 2, the rejection sampler uses an Exp(1) proposal:

```
draw Y ~ Exp(1)
if Y ≥ k: reject
draw V ~ Uniform(0, 1)
accept iff log V < (k-1)·log(1 - Y/k) + Y - log M_k
return X = 1 - Y/k
```

where `log M_k = (k-1) · log(1 - 1/k) + 1`. For k = 1, return U directly.

## Proof of correctness for the rejection sampler

**Setup.** The target density is `f_k(x) = k · x^(k-1)` on `[0, 1]`.
Apply the change of variables `Y = k(1 - X)`, so `X = 1 - Y/k` and
`dY = -k · dX`. The density of Y is

```
f_Y(y) = f_k(1 - y/k) · (1/k) = (1 - y/k)^(k-1)        for y ∈ [0, k].
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
`X = 1 - Y/k` is distributed as `f_k`. ∎

## Performance

**Watch out for autovectorization in microbenchmarks.** A naive benchmark
loop that accumulates samples will, on x86 with a SIMD libm, let LLVM emit
vectorized `pow` over runs of consecutive iterations — inflating apparent
`pow` throughput far above what a Cortex-M can achieve. The shipped
microbench wraps each call in `std::hint::black_box` to defeat this.

With `black_box` fences on a modern x86:

```
  k    pow (ns/sample)    rejection (ns/sample)    rej/pow
  2          26.0                  51.4              1.98x
  5          26.1                  45.7              1.75x
  10         26.1                  45.9              1.76x
  50         26.0                  36.7              1.41x
  200        26.2                  35.7              1.36x
  1000       26.1                  35.6              1.36x
```

Without `black_box`, the same machine reported `pow` at ~3 ns/sample —
nearly 10× faster. That number is real for vectorizable workloads but is
not representative of scalar per-call cost on hardware without SIMD.

The `pow` path wins on this host, but the gap is much smaller than the
unfenced numbers suggested. On a Cortex-M4F with a typical embedded libm,
where scalar `powf` may take 100+ cycles versus a `logf` of 30–50 cycles
plus an Exp(1) draw, the rejection path may win — particularly for moderate
to large k where the M_k overhead is small. **Benchmark on your actual
target before choosing.**

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

The crate provides the resampling algorithm that motivated the Beta(k, 1)
sampler in the first place.

```rust
use beta_k1::resample_indices;

let weights = vec![1.0, 3.0, 2.0, 4.0];
let mut out = vec![0usize; 1000];
resample_indices(&mut rng, &weights, &mut out);
// out now contains 1000 indices into `weights`, drawn with probability
// proportional to weight, sorted ascending.
```

`resample_indices` runs in O(`weights.len()` + `out.len()`) time. The
streaming sorted-uniforms generator is also exposed:

```rust
use beta_k1::SortedUniforms;
for u in SortedUniforms::new(&mut rng, 100) {
    // u_1 ≤ u_2 ≤ ... ≤ u_100, all in [0, 1)
}
```

The output indices from `resample_indices` come out in ascending order,
which is what most downstream code (e.g. gathering particles into a new
buffer) wants anyway. If you need them shuffled, do that as a post-pass.

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
`F(v) = 1 − (1 − v)ᵏ` on [0, 1], i.e., `min(V₁, ..., Vₖ) ~ Beta(1, k)`.
Equivalently, `1 − min(V₁, ..., Vₖ) ~ Beta(k, 1)` (it is distributed as
`max(V₁, ..., Vₖ)`, by the symmetry `Vᵢ ↔ 1 − Vᵢ`).

*Proof.* `Pr[min Vᵢ > v] = Pr[V₁ > v] · ... · Pr[Vₖ > v] = (1 − v)ᵏ`.       ∎

#### Theorem 1 (correctness of `SortedUniforms`)

The iterator `SortedUniforms::new(rng, n)` yields a sequence of values
distributed as the order statistics of n i.i.d. Uniform(0, 1) variates.

*Proof.* By induction on the iteration index `i ∈ {1, ..., n}`. Write
`lastᵢ` for the value of the internal `last` variable just after the i-th
yield, with `last₀ = 0`.

*Base case (i = 1).* On the first call, `remaining = n` and `last = 0`.
The iterator computes `spacing = 1 − beta_k_1(rng, n)`. By construction
of `beta_k_1`, this `spacing` is distributed as `Beta(1, n)`, which by
Lemma 2 is the distribution of the minimum of n i.i.d. Uniform(0, 1)
variates — i.e., the distribution of `U₍₁₎`. The yielded value is
`last + (1 − last) · spacing = spacing`, and `last₁ = spacing`. So
`last₁ ~ U₍₁₎`. ✓

*Inductive step.* Assume that after i yields, `(last₁, ..., lastᵢ)` has
the same joint distribution as `(U₍₁₎, ..., U₍ᵢ₎)`, the first i order
statistics of n i.i.d. Uniform(0, 1). The iterator now has
`remaining = n − i` and proceeds:

```
spacing = 1 − beta_k_1(rng, n − i)
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
`min(W₁, ..., Wₙ₋ᵢ) ~ Beta(1, n − i)`, which equals
`1 − beta_k_1(rng, n − i)` in distribution — exactly `spacing`.
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

*Proof.* `total` is computed by

```
total = 0; for w in weights: total += w
```

processing `weights` in index order. When `j` reaches `m − 1` in the
merge, `cumulative` has been built by initializing `cumulative = weights[0]`
and then executing `cumulative += weights[k]` for k = 1, 2, ..., m − 1
in order — visiting weights in the same order with the same accumulator
type (f64). IEEE 754 binary floating-point addition is deterministic and
is a function only of its operands and rounding mode (round-to-nearest
in Rust by default), so the two sequences yield bit-identical results:
`cumulative == total` exactly when `j = m − 1`.

Each value `u` yielded by `SortedUniforms` satisfies `u ≤ 1` (the
recurrence preserves this; see edge-case note below). Therefore
`target = total · u ≤ total = cumulative`, so the strict inequality
`target > cumulative` is false and the while loop exits with
`j = m − 1` rather than incrementing further.       ∎

**Edge case.** Strictly: if `beta_k_1(rng, k)` ever returns exactly 0
(possible for the `pow` backend if the underlying uniform draw is exactly
0, probability `2⁻⁵³` per draw), then `spacing = 1.0`, the recurrence
sets `last = 1.0`, and all subsequent yielded values equal 1. The merge
then sets `target = total` exactly; Lemma 3 still holds (the proof above
uses `u ≤ 1`, not the strict version). The statistical effect, however,
is that all subsequent index draws collapse to the last index, which is
incorrect. The probability of this occurring at least once during a
resampling pass of n outputs is bounded by `n·2⁻⁵³`, of order `10⁻¹⁰`
for n = 10⁶, far below the noise floor of any practical experiment.
The `rejection` backend is not subject to this edge case (its output
is strictly less than 1 by construction).

### Numerical robustness

Beyond the boundary argument: the merge accumulates `cumulative` in
left-to-right order, which is the standard direction for prefix sums
and is well-conditioned for non-negative summands. We do not use Kahan
summation; the relative error in `cumulative` is bounded by `O(m · ε)`
where ε ≈ 2.22·10⁻¹⁶, which is negligible compared to per-step
quantization error in the `f64` representation of weights and uniforms.

Producing exact bit-identical `total` and `cumulative` at the right
endpoint depends on the same accumulation order being used in both
places. This is guaranteed in the current implementation by both passes
iterating `weights` via `&[f64]` indexing in increasing index order;
changing either to a parallel reduce or a different traversal would
require revisiting Lemma 3.

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

Full-pipeline benchmark (m = n, weights = 1..=m, host-specific):

```
   m = n        ns/call   ns/step (m+n)
     100           4497           22.5
    1000          45339           22.7
   10000         452461           22.6
  100000        4488098           22.4
```

Linear scaling is clean across four orders of magnitude. The per-step
cost (~22 ns on the test machine, modern x86 with SIMD libm) is
dominated by the Beta(k, 1) draw and the merge's floating-point work;
cache effects don't materially degrade large-m performance because the
weight array is touched in a single forward sweep.

## Files

- `src/lib.rs` — both Beta(k,1) implementations, the `SortedUniforms`
  streaming iterator, and `resample_indices`.
- `src/main.rs` — test driver: KS tests, moment checks, chi-squared
  tests for resampling, fenced per-call microbench, full-pipeline bench.
- `Cargo.toml` — package manifest with feature gates.

## Building

```
# Default (pow):
cargo run --release --bin test_driver

# Rejection only:
cargo run --release --bin test_driver --no-default-features --features rejection
```
