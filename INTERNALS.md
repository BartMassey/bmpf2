# `ltsis` — Internals

Specification, design, implementation, and verification notes for the
`ltsis` crate. The user-facing `README.md` covers what the crate does and
how to call it; this document covers *how* and *why* it's built the way
it is. Read this if you're modifying the implementation, auditing the
math, porting to a different float type or platform, or considering
similar primitives in other crates.

---

## 1. Abstract

`ltsis` exposes O(n) primitives for **multinomial resampling** —
i.e. drawing n iid samples with replacement from a discrete weight
distribution. The construction combines the classical Bentley–Saxe
spacings recurrence for sorted uniforms with the merge-against-cumulative-
weights step of Massey (ICASSP 2008). Compared to that paper, the
present work supplies a corrected variate sampler (the original used a
buggy Ziggurat) and a careful floating-point boundary argument that
makes the merge memory-safe even at f32 precision. The crate is `f32`
throughout, allocation-free, and `no_std`-compatible.

---

## 2. Specification

### 2.1. Problem statement

Given a vector of `m` non-negative weights `w₁, ..., wₘ` with
`T = Σⱼ wⱼ > 0`, and an output count `n`, produce a sequence
`J₁ ≤ J₂ ≤ ... ≤ Jₙ` of indices into `weights` such that the joint
distribution of `(J₁, ..., Jₙ)` is identical to that of
`(K₍₁₎, ..., K₍ₙ₎)`, where `K₁, ..., Kₙ` are iid with
`Pr[Kₐ = j] = wⱼ / T` and `K₍·₎` denotes their order statistics.

In other words: equivalent to taking n iid multinomial draws on the
weight distribution and sorting them, but produced in a single O(m + n)
pass.

### 2.2. Public API

Two functions with identical signatures:

```rust
fn resample_indices         (rng: &mut impl Rng, weights: &[f32], out: &mut [u32]);
fn resample_indices_buffered(rng: &mut impl Rng, weights: &[f32], out: &mut [u32]);
```

Both write `out.len()` indices into `out`. Indices are `u32` rather
than `usize` so the on-disk layout is identical on every platform
(16-, 32-, or 64-bit `usize`).

Plus two lower-level primitives:

- `first_uniform(rng, k) -> f32` — samples `min(U₁, ..., Uₖ)` for
  k iid Uniform(0, 1). Equivalently, samples `Beta(1, k)`.
- `SortedUniforms::new(rng, n)` — an iterator yielding `n` Uniform(0, 1)
  variates in ascending order.

### 2.3. Preconditions

- `weights.is_empty() == false`.
- `weights.len() <= u32::MAX`.
- All weights are finite and non-negative (debug-asserted).
- `Σ weights > 0`.

Note that the "memory-safe" guarantee rests on Lemma 3
(§5.2).

---

## 3. Design

### 3.1. The order-statistic recurrence

The sorted uniforms `U₍₁₎ < U₍₂₎ < ... < U₍ₙ₎` of `n` iid
Uniform(0, 1) draws can be generated sequentially in O(n) via the
spacings recurrence (Bentley & Saxe 1980; Devroye 1986, §V.3.1). At
step `i` with `last = U₍ᵢ₋₁₎`:

```
spacing  ~  Beta(1, k)         where k = n − i + 1
yield    =  last + (1 − last) · spacing
```

The conditional distribution of `U₍ᵢ₎` given `U₍ᵢ₋₁₎ = u` is the
minimum of `n − i + 1` iid Uniform(u, 1) draws (Lemma 1, §5.1), and
that minimum equals `u + (1 − u) · Z` where `Z` is the minimum of
`n − i + 1` iid Uniform(0, 1) draws — i.e. `Z ~ Beta(1, k)`.
[`SortedUniforms`] implements this recurrence; [`first_uniform`]
samples `Z`.

### 3.2. The merge

Given sorted uniforms and a cumulative weight array
`Wⱼ = w₁ + ... + wⱼ` (so `Wₘ = T`), the inverse-CDF construction of
multinomial sampling (Theorem 2, §5.1) gives

```
Jᵢ  =  min { j : T · U₍ᵢ₎ < Wⱼ }.
```

Because `j ↦ Wⱼ` is monotone, the map `U ↦ J` is monotone non-
decreasing; sorting the inputs gives sorted outputs. Implementation:
walk the sorted uniforms left-to-right with a single advancing
cursor `j`, maintaining `cumulative = Wⱼ` as a running prefix sum.
For each yielded `U₍ᵢ₎`, advance `j` while `total · U₍ᵢ₎ > cumulative`,
then record `Jᵢ = j`. Total work O(m + n) since each cursor advances
monotonically.

### 3.3. f32 throughout

All public APIs are `f32`. The realistic deployment target is
Cortex-M4F (and similar single-precision FPUs), where every f64
operation is a software call. Going all-f32 keeps the hot path on
the FPU; numerical robustness comes from Kahan summation (§4.4),
not from extra precision.

### 3.4. Two variants

- **`resample_indices`** (streaming): generates each sorted uniform
  via [`SortedUniforms`] / [`first_uniform`] — one `powf` per output
  index. No additional memory beyond `out`.
- **`resample_indices_buffered`** (buffered): uses a different
  sorted-uniforms generator — the Gamma-ratio identity
  `U₍ᵢ₎ = (E₁ + ... + Eᵢ) / (E₁ + ... + Eₙ₊₁)` where `Eⱼ ~ Exp(1)`
  iid. Trades one Exp(1) draw per output for the `powf`. On x86 with
  a tuned libm this is ~1.3× faster per element; on Cortex-M4F where
  scalar `powf` is much slower than an Exp Ziggurat, the gap widens.

Identical signatures (no caller-supplied scratch); the buffered
variant repurposes `out` as scratch via [`f32::to_bits`] (§4.5).

---

## 4. Implementation

### 4.1. `first_uniform` — inverse CDF in f32

The minimum of `k` iid Uniform(0, 1) draws has CDF
`F(x) = 1 − (1 − x)^k` and inverse `F⁻¹(u) = 1 − (1 − u)^(1/k)`.
Implementation:

```rust
let u: f32 = rng.gen();   // u ∈ [0, 1 − 2⁻²⁴]
1.0 - (1.0 - u).powf(1.0 / k as f32)
```

This form is preferred over the algebraically equivalent
`1 − u^(1/k)` (from substituting `v = 1 − u`, also uniform) because
it has better f32 boundary behavior. With `1 − u^(1/k)`, the input
`u = 0` (which `rng.gen()` produces with probability 2⁻²⁴) yields
`0^(1/k) = 0`, and the function returns 1 — outside the
`[0, 1)` support, which would freeze the order-statistic recurrence
at `last = 1`. Earlier versions guarded this with a redraw.

The chosen form `1 − (1 − u)^(1/k)` is well-behaved with no special
case: `1 − u` lands in `[2⁻²⁴, 1]` exactly representably in f32
(since `1 − i · 2⁻²⁴` is f32-representable for `i = 0..2²⁴`), so
`(1 − u)^(1/k) ∈ [2⁻²⁴ᐟᵏ, 1]` and the output is in
`[0, 1 − 2⁻²⁴ᐟᵏ] ⊂ [0, 1)`. Each of the 2²⁴ input bins maps to a
distinct output, all in range.

There is one benign rounding artifact: for very large `k` and `u`
near 0, `(1 − u)^(1/k)` can round to exactly 1 in f32, making the
output 0 and `spacing = 0`; the recurrence then yields the prior
`last` again. This is an f32 quantization of "consecutive order
statistics rounded to the same value" and not a Lemma 3 violation.

### 4.2. `SortedUniforms` iterator

State: `(rng: &mut R, remaining: u32, last: f32)`. Each `next()`:

1. If `remaining == 0`, return `None`.
2. `let spacing = first_uniform(rng, remaining);`
3. `last += (1.0 - last) * spacing;`
4. `remaining -= 1;`
5. `Some(last)`.

Implements `Iterator`, `ExactSizeIterator` (`size_hint` returns
`(remaining, Some(remaining))` exactly), and `FusedIterator` (once
`remaining` hits 0, all further calls return `None`). `count()` is
overridden to return `remaining as usize` directly without
exhausting the RNG.

### 4.3. `resample_indices` (streaming)

```
1. Kahan-sum `total = Σ weights`.
2. For each of n output slots:
   a. Pull a sorted uniform `u` from SortedUniforms.
   b. Compute `target = total * u`.
   c. While `target > cumulative`:
        j += 1; Kahan-add weights[j] to cumulative.
   d. Write out[i] = j as u32.
```

`cumulative` starts at `weights[0]` (equivalent to one Kahan step
from `(0, 0)`). The `total` walk and the merge's `cumulative` walk
are both Kahan sums of `weights` in index order from `(sum, c) = (0, 0)`,
so they end bit-for-bit equal — a precondition of Lemma 3.

### 4.4. `resample_indices_buffered` (buffered)

```
1. Kahan-sum `total = Σ weights`.
2. Phase 1: for each of n output slots:
   a. Draw `e ~ Exp(1)` (Ziggurat).
   b. Stash `e.to_bits()` into out[i] (treating the u32 slot as f32 bits).
   c. Kahan-add e to running G.
3. Draw e_extra ~ Exp(1); Kahan-add to G (don't store).
4. Compute inv_g = 1/G.
5. Phase 2: walk out left-to-right:
   a. e = f32::from_bits(out[i]); recover the stashed Exp draw.
   b. Kahan-add e to cumulative_e (= S_i).
   c. u = cumulative_e * inv_g; target = (total * u).min(total).
   d. While target > cumulative_w:
        j += 1; Kahan-add weights[j] to cumulative_w.
   e. Write out[i] = j as u32.
```

Note that this routine does *not* call `first_uniform` or use
`SortedUniforms` — the Gamma-ratio identity gives sorted uniforms
directly without `powf`.

The `target.min(total)` clip handles the rare f32 case where
`u_n = S_n / G` rounds up to exactly 1.0 (in f32 this happens when
`E_{n+1}/G < 2⁻²⁵`, with probability ~3% at `n = 10⁶`). Without the
clip, `target` could exceed `cumulative_w` at the right endpoint
even though `u < 1` in exact arithmetic; with it, the merge loop is
guaranteed to terminate within `weights.len()`.

### 4.5. In-place scratch via `f32::to_bits` round-trip

The buffered variant needs `n` Exp(1) draws temporarily before it
walks them again to compute indices. Rather than asking the caller
for a `&mut [f32]` scratch buffer, it stashes each draw's bit
pattern in the output slot it'll later overwrite with the index:

```rust
// Phase 1 store:
*slot = e.to_bits();          // u32, exact bit representation of e

// Phase 2 load:
let e = f32::from_bits(*slot);
// ... compute index j ...
*slot = j as u32;
```

`to_bits` / `from_bits` round-trips are exact for every f32 value
(including NaN and ±∞). Because `out` slots are exactly 32 bits
(`u32`, not `usize`), the layout is platform-independent.

### 4.6. Kahan compensated summation

Every multi-term accumulator on the data path uses Kahan summation:

```rust
fn kahan_add(sum: &mut f32, c: &mut f32, x: f32) {
    let y = x - *c;
    let t = *sum + y;
    *c = (t - *sum) - y;
    *sum = t;
}
```

Naive f32 prefix sums over n non-negative terms accrue relative
error of order `O(n · 2⁻²⁴)`, which becomes unusable around
n ≈ 10⁵. Kahan reduces the bound to `O(2⁻²⁴ · max|term|)` —
effectively constant in `n`.

Accumulators using Kahan:
- `total = Σ weights` and the merge's incremental `cumulative_w`
  in both `resample_indices` and `resample_indices_buffered`.
- `G = Σ Eᵢ` and the merge's `cumulative_e = Sᵢ` in
  `resample_indices_buffered`.

The bit-for-bit identity used in Lemma 3 depends on both the
up-front `total` walk and the merge's `cumulative_w` walk using the
same algorithm and traversal order. Both Kahan-sum `weights` in
increasing index order from `(sum, c) = (0, 0)`. Any change here —
parallel reduce, different traversal, switching to a non-Kahan
accumulator — would invalidate Lemma 3.

---

## 5. Validation & verification

(This is Claude output, currently unchecked by a human
statistician. Corrections welcome.)

### 5.1. Mathematical correctness

#### Lemma 1 (memorylessness of uniform order statistics)

Let `U₁, ..., Uₙ` be iid Uniform(0, 1) and let
`U₍₁₎ ≤ ... ≤ U₍ₙ₎` be their order statistics. For any
1 ≤ i ≤ n − 1, conditional on `U₍ᵢ₎ = u`, the remaining order
statistics `U₍ᵢ₊₁₎, ..., U₍ₙ₎` are jointly distributed as the order
statistics of n − i iid Uniform(u, 1) variates.

*Proof.* Standard property of order statistics from a continuous
distribution; see Devroye (1986), §V.3, or David & Nagaraja (2003),
§2.4. The key fact: conditional on `U₍ᵢ₎ = u`, the values `Uⱼ`
exceeding `u` are iid Uniform(u, 1). ∎

#### Lemma 2 (minimum of k uniforms)

If `V₁, ..., Vₖ` are iid Uniform(0, 1) then `min(V₁, ..., Vₖ)` has
CDF `F(v) = 1 − (1 − v)ᵏ` on `[0, 1]` — i.e.
`min(V₁, ..., Vₖ) ~ Beta(1, k)`.

*Proof.* `Pr[min Vᵢ > v] = Πᵢ Pr[Vᵢ > v] = (1 − v)ᵏ`. ∎

#### Theorem 1 (correctness of `SortedUniforms`)

The iterator `SortedUniforms::new(rng, n)` yields a sequence of
values distributed as the order statistics of n iid Uniform(0, 1)
variates.

*Proof.* By induction on `i ∈ {1, ..., n}`. Write `lastᵢ` for the
internal `last` after the i-th yield, with `last₀ = 0`.

*Base case (i = 1).* `remaining = n`, `last = 0`. Compute
`spacing = first_uniform(rng, n)`, distributed as Beta(1, n) by
construction; by Lemma 2 this is the distribution of the minimum
of n iid Uniform(0, 1) — i.e. of `U₍₁₎`. Yield is
`0 + 1 · spacing = spacing`, so `last₁ ~ U₍₁₎`. ✓

*Inductive step.* Assume `(last₁, ..., lastᵢ)` has the joint
distribution of `(U₍₁₎, ..., U₍ᵢ₎)`. Now `remaining = n − i`,
`spacing = first_uniform(rng, n − i) ~ Beta(1, n − i)`. By Lemma 1,
conditional on `lastᵢ = u`, `U₍ᵢ₊₁₎` is the minimum of n − i iid
Uniform(u, 1) draws, equal in distribution to
`u + (1 − u) · min(W₁, ..., Wₙ₋ᵢ)` for `Wⱼ` iid Uniform(0, 1). By
Lemma 2 the inner min is Beta(1, n − i), exactly the distribution
of `spacing`. So `lastᵢ₊₁ = lastᵢ + (1 − lastᵢ) · spacing` has the
correct conditional distribution given `lastᵢ`, extending the
hypothesis to step i + 1. ∎

#### Theorem 2 (correctness of `resample_indices`)

Let `w₁, ..., wₘ ≥ 0` with `T = Σⱼ wⱼ > 0`. Define iid multinomial
draws `K₁, ..., Kₙ` with `Pr[Kₐ = j] = wⱼ / T`. Then the output
sequence `J₁ ≤ ... ≤ Jₙ` produced by
`resample_indices(rng, weights, out)` (with `out.len() = n`) has
the same joint distribution as the sorted multinomial sample
`(K₍₁₎, ..., K₍ₙ₎)`. (1-indexed in this proof; the code is
0-indexed.)

*Proof.* Let `Wⱼ = w₁ + ... + wⱼ`, `F(j) = Wⱼ / T`. The inverse-CDF
multinomial sampler draws `Uₐ ~ Uniform(0, 1)` and sets

```
Kₐ = min { j : F(j) > Uₐ } = min { j : T · Uₐ < Wⱼ }.       (∗)
```

This is correct because `F(j − 1) ≤ Uₐ < F(j)` happens with
probability `wⱼ / T`. The map `Uₐ ↦ Kₐ` is monotone non-decreasing,
so sorting the `Uₐ` and applying (∗) yields the sorted multinomial
sample:

```
(K₍₁₎, ..., K₍ₙ₎) = (φ(U₍₁₎), ..., φ(U₍ₙ₎))                  (∗∗)
```

where `φ(u) = min { j : T · u < Wⱼ }`. By Theorem 1, `resample_indices`
yields `(U₍₁₎, ..., U₍ₙ₎)` distributed as the order statistics of n
iid Uniform(0, 1). For each yielded `U₍ᵢ₎`, the merge sets
`target = T · U₍ᵢ₎` and advances `j` until `target ≤ cumulative = Wⱼ`,
giving

```
Jᵢ = min { j : Wⱼ ≥ T · U₍ᵢ₎ }.                              (∗∗∗)
```

Predicates (∗) and (∗∗∗) differ only on the event `T · U = Wⱼ`,
a measure-zero event under the continuous uniform distribution. So
`Jᵢ = φ(U₍ᵢ₎)` almost surely, matching (∗∗). ∎

### 5.2. Lemma 3 (floating-point boundary)

Suppose `weights[i]` are finite and non-negative with positive sum.
Then `resample_indices` cannot index past `weights.len() − 1`
regardless of values yielded by `SortedUniforms`, provided those
values lie in `[0, 1]`.

*Proof.* `total` is computed by Kahan compensated summation walking
`weights` in index order from `(sum, c) = (0, 0)`:

```
(total, total_c) = (0, 0)
for w in weights:
    (total, total_c) = kahan_add(total, total_c, w)
```

When `j` reaches `m − 1` in the merge, `(cumulative, cumulative_c)`
has been built by initializing `(weights[0], 0)` (equivalent to one
Kahan step from `(0, 0)`, since the first compensator update is
zero) then `kahan_add(cumulative, cumulative_c, weights[k])` for
`k = 1, ..., m − 1` — same weights, same order, same accumulator
algorithm. IEEE 754 addition is deterministic in operands and
rounding mode (round-to-nearest, default in Rust), so Kahan
summation is also deterministic. The two sequences yield bit-
identical results: `cumulative == total` exactly when `j = m − 1`.

Each value `u` yielded by `SortedUniforms` satisfies `u ≤ 1`. Since
`total` is f32-representable and `u ∈ [0, 1]`, the exact product
`total · u ≤ total`; in round-to-nearest f32 this rounds to a
representable value `≤ total` (because `total` itself is
representable, every value `< total` rounds to a representable
value `≤ total`). So `target ≤ total = cumulative` whenever
`j = m − 1`, and the strict inequality `target > cumulative` is
false: the while loop exits with `j = m − 1` rather than
incrementing further. ∎

The buffered variant additionally applies a `target.min(total)`
clip in its merge. This handles the rare f32 case where
`u_n = S_n / G` rounds up to exactly 1.0 (the underlying inputs
satisfy `S_n < G` strictly, but f32 rounding can lose the strict
inequality). Without the clip, `target` could nominally exceed
`total = cumulative_w` at the right endpoint; with it, the merge
terminates within `weights.len()` regardless.

### 5.3. f32 quantization edge case

`first_uniform` returns values strictly in `[0, 1)` by
construction (§4.1). Each of the 2²⁴ input bins from
`rng.gen::<f32>()` maps to a distinct output. For very large `k`
and `u` near 0, `(1 − u)^(1/k)` can round to exactly 1 in f32,
making the output 0 and `spacing = 0`. The recurrence then yields
the prior `last` again — a vanishing statistical artifact (the
f32 quantization of "consecutive order statistics rounded to the
same value"), not a Lemma 3 violation since `last ≤ 1` is
preserved.

### 5.4. Statistical test methodology

(These statistical tests were generated by Claude, and have
not been validated by any human mathematician. Corrections
welcome.)

The integration tests in `tests/statistical.rs` (run via
`cargo test`) check the following invariants. All RNG seeds are
fixed; thresholds are calibrated to keep the **aggregate
random-failure probability under correct code below 1e-9** (the
ten tests share an aggregate budget of `1e-9`, so each test
reserves `~1e-10` and each sub-check inside a test reserves
`~1e-11` after Bonferroni correction over its sub-checks).

1. **Range check** (`range`, deterministic): every `first_uniform`
   sample lies in `[0, 1)` across `k ∈ {1, 2, 3, 5, 10, 100, 1000}`,
   100k samples each.
2. **Empirical moments** (`moments_first_uniform`): mean and
   variance of `first_uniform(k)` match the closed-form Beta(1, k)
   values across `k ∈ {1, 2, 3, 5, 10, 50, 200}`, 1M samples each.
   Mean tolerance 7σ; variance tolerance 2% relative (≈ 14σ at
   1M samples).
3. **One-sample KS** (`ks_against_theory`): empirical CDF of
   `first_uniform` samples vs. analytic `Fₖ(x) = 1 − (1 − x)ᵏ`,
   50k samples each. Critical-value coefficient `c = 3.7`
   (≈ `2·exp(−2c²) ≈ 2.5e-12` per sub-check).
4. **Two-sample KS** (`ks_against_min_oracle`): `first_uniform`
   vs. an independent min-of-k-uniforms oracle (literally
   `min(rng.gen::<f32>(), ...)` over `k` draws), 20k samples each,
   `c = 3.7`.
5. **Sorted-uniforms per-position moments** (`sorted_uniforms_moments`):
   empirical mean and variance at each position match the closed-form
   order-statistic values, 200k runs. Per-position mean tolerance
   `7.5σ` (covers `n` up to 100 positions per test).
6. **Pooled sorted-uniforms KS** (`sorted_uniforms_pooled_ks`):
   pooled output vs. uniform CDF, `c = 3.7`.
7. **Resampler marginal χ²** (`resample_marginals_streaming` and
   `resample_marginals_buffered`): per-index frequency under each
   resampler matches the weight-proportional probabilities, four
   weight cases (uniform, decreasing, peaky, with-zeros).
   Wilson–Hilferty z = 7.0.
8. **Resampler vs. naive multinomial χ²**
   (`resample_vs_multinomial_streaming` and
   `resample_vs_multinomial_buffered`): two-sample chi-squared on
   index-count vectors comparing each resampler against an
   O(m + n log m) inverse-CDF naive multinomial reference (run in
   f64 internally so the reference has no prefix-sum noise of its
   own). Wilson–Hilferty z = 7.0.

The 1e-9 budget is per-fixed-`rand`-version and with fixed
variate usage by the code. If `rand` bumps the algorithm of
any of its samplers, or if the code usage of `rand` changes,
the seeds will produce different sequences and the tests
will need to be re-run. `Cargo.lock` is committed precisely
to pin the `rand` version against this risk in CI /
development.

### 5.5. Microbenchmark methodology

Per-call measurements wrap each call in `std::hint::black_box` to
defeat cross-iteration LLVM autovectorization. A naive loop on x86
with a SIMD libm would let LLVM emit a vectorized `powf` over runs
of consecutive iterations — measuring batched throughput rather
than scalar per-call cost. On Cortex-M4F (and any other no-SIMD
target) batched throughput is unattainable, so scalar per-call
cost is the relevant number. The fences pin the measurement to
that.

Note that *internal* libm SIMD use (using SIMD instructions
to compute one scalar `powf` more quickly) is unaffected by
`black_box`. The 9 ns/call we measure on x86 is the genuine
cost of one scalar `f32::powf` on the host's libm.

---

## 6. Performance

### 6.1. `first_uniform` per-call cost (host x86, `black_box`-fenced)

| k | ns/sample |
|----|----|
| 2  | 9.9 |
| 5  | 10.1 |
| 10 | 10.0 |
| 50 | 10.1 |
| 200 | 9.9 |
| 1000 | 10.0 |

Essentially flat in `k`. Dominated by the scalar `f32::powf` call;
the RNG (Xoshiro256++ in `SmallRng`) is sub-nanosecond per draw and
contributes ~1 ns at most.

### 6.2. Full resampling pipeline

`m = n`, weights `1..=m`, `f32`, all calls `black_box`-fenced,
SmallRng (Xoshiro256++) seed-based. Run-to-run variance is ~3% on
each cell.

```
    m = n     streaming                buffered              streaming
              ns/call    ns/step       ns/call    ns/step    /buffered
      100         2870     14.35          2161     10.81        1.33x
     1000        29184     14.59         21862     10.93        1.33x
    10000       284775     14.24        222624     11.13        1.28x
   100000      2892905     14.46       2193198     10.97        1.32x
  1000000     28895398     14.45      21832852     10.92        1.32x
```

Linear scaling clean across five orders of magnitude. Per-step
cost dominated by the `Beta(1, k)` draw (streaming) or the
`Exp(1)` draw plus merge (buffered); cache effects don't
materially degrade large-`m` performance because the weight array
is touched in a single forward sweep.

### 6.3. Cortex-M4F expectations

On Cortex-M4F (no SIMD, scalar `powf` ~100–150 cycles, `Exp(1)`
Ziggurat ~30–60 cycles), the buffered variant is expected to win
by a larger margin than the ~1.32× we see on x86. We have no on-
target measurements yet.

---

## 7. Future directions

### 7.1. Padé[m/m] rational squeeze for `first_uniform`

The library currently spends one `powf` per `first_uniform` call.
On Cortex-M4F that's expensive. An earlier exploration around a
polynomial squeeze was buggy (it ignored the Mₖ normalization in a
rejection-sampler formulation that was being tried at the time and
produced biased samples — empirical mean for k=2 came out 0.675 vs
theoretical 0.667). The original ICASSP 2008 paper's Ziggurat had a
similar normalization issue (envelope geometry mismatched across
n). Neither is retained.

A more promising not-yet-implemented direction is a Padé[m/m]
rational squeeze in a rejection scheme. The Padé[1/1] form of
`(1 − u)^(1/k)` reduces strikingly cleanly:

```
(1 − u)^(1/k)  ≈  (2k − (k+1)u) / (2k − (k−1)u)
```

so the spacing approximates `u / (k − (k−1)u/2)` (one mult, one
subtract, one divide; no transcendentals). Used directly as a
sampler this is biased — error grows large near `u → 1`, the rare-
but-real "large spacing" tail — and the bias compounds in
`SortedUniforms` over `n = 10⁶` samples. Used as a one-sided
*squeeze* in a rejection scheme with `powf` on the slow path, the
cost amortizes: most attempts decide via the cheap rational test,
and the rare `powf` evaluation is statistically negligible at
moderate-to-large `k`.

On host x86, this is unlikely to beat scalar `powf` (a tuned
libm at ~9 ns/call is hard to beat). On Cortex-M4F, where scalar
`powf` is ~100–150 cycles, an all-mults-and-adds fast path could
plausibly win 30–50%. Whether that's worth the complexity
(correctness proof for one-sidedness, careful slow-path semantics)
depends on the deployment target.

---

## 8. References

- Bentley, J. L. & Saxe, J. B. (1980). Generating sorted lists of
  random numbers. *ACM Transactions on Mathematical Software*,
  6(3), 359–364. Original linear-time sequential generator for
  sorted uniforms.
- David, H. A. & Nagaraja, H. N. (2003). *Order Statistics*, 3rd
  ed. Wiley. Comprehensive treatment of joint distribution of
  order statistics.
- Devroye, L. (1986). *Non-Uniform Random Variate Generation.*
  Springer. Chapter V (Uniform and Exponential Spacings); §V.3.1
  treats generation of uniform `[0, 1]` order statistics.
  Available at https://luc.devroye.org/rnbookindex.html.
- Massey, B. (2008). Fast perfect weighted resampling.
  *Proceedings of IEEE ICASSP 2008*. The merge-with-sorted-uniforms
  construction this crate implements.
