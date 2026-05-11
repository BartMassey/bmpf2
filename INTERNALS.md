# `ltsis` — Internals

Specification, design, implementation, and verification notes for the
`ltsis` crate. The user-facing `README.md` covers what the crate does and
how to call it; this document covers *how* and *why* it's built the way
it is. Read this if you're modifying the implementation, auditing the
math, porting to a different float type or platform, or considering
similar primitives in other crates.

---

## 1. Abstract

`ltsis` exposes O(n) primitives for **multinomial sampling** —
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

Given a vector of $m$ non-negative weights $w_1, \ldots, w_m$ with
$T = \sum_j w_j > 0$, and an output count $n$, produce a sequence
$J_1 \le J_2 \le \cdots \le J_n$ of indices into `weights` such that
the joint distribution of $(J_1, \ldots, J_n)$ is identical to that
of $(K_{(1)}, \ldots, K_{(n)})$, where $K_1, \ldots, K_n$ are iid
with $\Pr[K_a = j] = w_j / T$ and $K_{(\cdot)}$ denotes their order
statistics.

In other words: equivalent to taking $n$ iid multinomial draws on
the weight distribution and sorting them, but produced in a single
$O(m + n)$ pass.

### 2.2. Public API

Two functions, deliberately not symmetric:

```rust
fn sample_indices<'a, R>(rng: &'a mut R, weights: &'a [f32], n: u32)
    -> SampleIndices<'a, R>;                   // yields n × u32

fn sample_indices_buffered(rng: &mut impl Rng,
                           weights: &[f32],
                           out: &mut [u32]);
```

`sample_indices` returns an iterator that yields `n` `u32` indices
in ascending order. The buffered variant takes an `&mut [u32]`
instead — it uses each output slot as f32 scratch (via
`f32::to_bits`/`from_bits`, see §4.5), so it cannot be lazy.

Indices are `u32` rather than `usize` so the on-disk layout is
identical on every platform (16-, 32-, or 64-bit `usize`).

Plus two lower-level primitives:

- `first_uniform(rng, k) -> f32` — samples
  $\min(U_1, \ldots, U_k)$ for $k$ iid $\mathrm{Uniform}(0, 1)$.
  Equivalently, samples $\mathrm{Beta}(1, k)$.
- `SortedUniforms::new(rng, n)` — an iterator yielding $n$
  $\mathrm{Uniform}(0, 1)$ variates in ascending order.

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

The sorted uniforms $U_{(1)} < U_{(2)} < \cdots < U_{(n)}$ of $n$
iid $\mathrm{Uniform}(0, 1)$ draws can be generated sequentially in
$O(n)$ via the spacings recurrence (Bentley & Saxe 1980;
Devroye 1986, §V.3.1). At step $i$ with $\mathrm{last} = U_{(i-1)}$:

$$
\mathrm{spacing} \sim \mathrm{Beta}(1, k), \qquad k = n - i + 1
$$
$$
\mathrm{yield} = \mathrm{last} + (1 - \mathrm{last}) \cdot \mathrm{spacing}
$$

The conditional distribution of $U_{(i)}$ given
$U_{(i-1)} = u$ is the minimum of $n - i + 1$ iid
$\mathrm{Uniform}(u, 1)$ draws (Lemma 1, §5.1), and that minimum
equals $u + (1 - u) \cdot Z$ where $Z$ is the minimum of
$n - i + 1$ iid $\mathrm{Uniform}(0, 1)$ draws — i.e.
$Z \sim \mathrm{Beta}(1, k)$. [`SortedUniforms`] implements this
recurrence; [`first_uniform`] samples $Z$.

### 3.2. The merge

Given sorted uniforms and a cumulative weight array
$W_j = w_1 + \cdots + w_j$ (so $W_m = T$), the inverse-CDF
construction of multinomial sampling (Theorem 2, §5.1) gives

$$
J_i = \min \{\, j : T \cdot U_{(i)} < W_j \,\}.
$$

Because $j \mapsto W_j$ is monotone, the map $U \mapsto J$ is
monotone non-decreasing; sorting the inputs gives sorted outputs.
Implementation: walk the sorted uniforms left-to-right with a single
advancing cursor $j$, maintaining $\mathrm{cumulative} = W_j$ as a
running prefix sum. For each yielded $U_{(i)}$, advance $j$ while
$\mathrm{total} \cdot U_{(i)} > \mathrm{cumulative}$, then record
$J_i = j$. Total work $O(m + n)$ since each cursor advances
monotonically.

### 3.3. f32 throughout

All public APIs are `f32`. The realistic deployment target is
Cortex-M4F (and similar single-precision FPUs), where every f64
operation is a software call. Going all-f32 keeps the hot path on
the FPU; numerical robustness comes from Kahan summation (§4.4),
not from extra precision.

### 3.4. Two variants

- **`sample_indices`** (streaming): generates each sorted uniform
  via [`SortedUniforms`] / [`first_uniform`] — one `powf` per output
  index. Returns an iterator; no additional memory.
- **`sample_indices_buffered`** (buffered): uses a different
  sorted-uniforms generator — the Gamma-ratio identity
  $U_{(i)} = (E_1 + \cdots + E_i) / (E_1 + \cdots + E_{n+1})$
  where $E_j \sim \mathrm{Exp}(1)$ iid. Trades one
  $\mathrm{Exp}(1)$ draw per output for the `powf`. On x86 with a
  tuned libm this is ~1.3× faster per element; on Cortex-M4F where
  scalar `powf` is much slower than an Exp Ziggurat, the gap widens.

The buffered variant repurposes the caller's `out` slice as f32
scratch via [`f32::to_bits`] (§4.5), so it cannot expose an
iterator; the streaming variant has no scratch to share and so is
free to.

---

## 4. Implementation

### 4.1. `first_uniform` — inverse CDF in f32

The minimum of $k$ iid $\mathrm{Uniform}(0, 1)$ draws has CDF
$F(x) = 1 - (1 - x)^k$ and inverse
$F^{-1}(u) = 1 - (1 - u)^{1/k}$. Implementation:

```rust
let u: f32 = rng.gen();   // u ∈ [0, 1 − 2⁻²⁴]
1.0 - (1.0 - u).powf(1.0 / k as f32)
```

This form is preferred over the algebraically equivalent
$1 - u^{1/k}$ (from substituting $v = 1 - u$, also uniform)
because it has better f32 boundary behavior. With $1 - u^{1/k}$,
the input $u = 0$ (which `rng.gen()` produces with probability
$2^{-24}$) yields $0^{1/k} = 0$, and the function returns 1 —
outside the $[0, 1)$ support, which would freeze the order-
statistic recurrence at $\mathrm{last} = 1$. Earlier versions
guarded this with a redraw.

The chosen form $1 - (1 - u)^{1/k}$ is well-behaved with no
special case: $1 - u$ lands in $[2^{-24}, 1]$ exactly representably
in f32 (since $1 - i \cdot 2^{-24}$ is f32-representable for
$i = 0, \ldots, 2^{24}$), so
$(1 - u)^{1/k} \in [2^{-24/k}, 1]$ and the output is in
$[0, 1 - 2^{-24/k}] \subset [0, 1)$. Each of the $2^{24}$ input
bins maps to a distinct output, all in range.

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

### 4.3. `sample_indices` (streaming)

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

### 4.4. `sample_indices_buffered` (buffered)

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

The `target.min(total)` clip is load-bearing for memory safety
because of the multiply-by-inverse design choice (step 4 / step 5c).
We compute `inv_g = 1.0 / G` once and then `u = cumulative_e *
inv_g` per output, rather than `u = cumulative_e / G`. This saves
one f32 division per element on the hot path, but introduces an
asymmetry: while `cumulative_e ≤ G` strictly in exact arithmetic
(both Kahan-summed in the same order, with `G` including the extra
`E_{n+1} > 0`), in f32 the rounded inverse `inv_g` can sit slightly
above `1/G`, and then `cumulative_e * inv_g` can round to a value
just above 1.0 — namely `1.0 + 2⁻²³`, the next f32 above 1.0.
Without the clip, `target = total * u` could then strictly exceed
`cumulative_w = total` at the right endpoint, and the merge would
advance `j` past `weights.len() - 1`.

With the clip, `target ≤ total`, and Lemma 3 (§5.2) — which gives
`cumulative_w == total` bit-for-bit at `j = m − 1` — keeps the
merge inside the slice exactly as it does for the streaming
variant.

(An alternative that avoids the clip entirely would be to compute
`u = cumulative_e / G` directly. The natural-bound argument from
streaming then transfers verbatim. We don't take that route because
the per-element division is measurably slower than multiply-by-
precomputed-inverse on every CPU we care about.)

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
  in both `sample_indices` and `sample_indices_buffered`.
- `G = Σ Eᵢ` and the merge's `cumulative_e = Sᵢ` in
  `sample_indices_buffered`.

The bit-for-bit identity used in Lemma 3 depends on both the
up-front `total` walk and the merge's `cumulative_w` walk using the
same algorithm and traversal order. Both Kahan-sum `weights` in
increasing index order from `(sum, c) = (0, 0)`. Any change here —
parallel reduce, different traversal, switching to a non-Kahan
accumulator — would invalidate Lemma 3.

### 4.7. Precondition checks in release builds

All four preconditions (§2.3) are checked with `assert!` in
release, not `debug_assert!`:

1. `weights` nonempty — O(1).
2. `weights.len() ≤ u32::MAX` — O(1).
3. Each `weights[i]` finite and nonnegative — O(m), one extra
   branch per element inside the `total` Kahan loop.
4. `Σ weights > 0` — O(1) (a single comparison after the sum).

Items 1, 2, 4 are clearly free. The interesting one is item 3:
its cost is amortized into a loop that already does a Kahan add
(four f32 ops per element). Microbenchmark on the host (m = n =
10⁶, weights `1..=m`) shows the full pipeline shifting from
15.0 ns/step to 15.3 ns/step (streaming) and 11.2 → 11.4 ns/step
(buffered) — about 2% slowdown each, well under the 5% bar we
set when deciding whether to keep this check release-on. We
keep it on because the cost of *not* checking — silent garbage
output from a stray NaN, or an out-of-bounds index from a
negative weight that smuggles `cumulative_w` past `total` —
is much worse than a couple of cycles per element on the cold
prefix-sum loop.

The merge loop itself does no such checks: it operates on
`weights[j]` after we've already validated the slice once.

---

## 5. Validation & verification

(This is Claude output, currently unchecked by a human
statistician. Corrections welcome.)

### 5.1. Mathematical correctness

#### Lemma 1 (memorylessness of uniform order statistics)

Let $U_1, \ldots, U_n$ be iid $\mathrm{Uniform}(0, 1)$ and let
$U_{(1)} \le \cdots \le U_{(n)}$ be their order statistics. For any
$1 \le i \le n - 1$, conditional on $U_{(i)} = u$, the remaining
order statistics $U_{(i+1)}, \ldots, U_{(n)}$ are jointly
distributed as the order statistics of $n - i$ iid
$\mathrm{Uniform}(u, 1)$ variates.

*Proof.* Standard property of order statistics from a continuous
distribution; see Devroye (1986), §V.3, or David & Nagaraja (2003),
§2.4. The key fact: conditional on $U_{(i)} = u$, the values $U_j$
exceeding $u$ are iid $\mathrm{Uniform}(u, 1)$. ∎

#### Lemma 2 (minimum of k uniforms)

If $V_1, \ldots, V_k$ are iid $\mathrm{Uniform}(0, 1)$ then
$\min(V_1, \ldots, V_k)$ has CDF $F(v) = 1 - (1 - v)^k$ on
$[0, 1]$ — i.e. $\min(V_1, \ldots, V_k) \sim \mathrm{Beta}(1, k)$.

*Proof.*
$\Pr[\min_i V_i > v] = \prod_i \Pr[V_i > v] = (1 - v)^k$. ∎

#### Theorem 1 (correctness of `SortedUniforms`)

The iterator `SortedUniforms::new(rng, n)` yields a sequence of
values distributed as the order statistics of $n$ iid
$\mathrm{Uniform}(0, 1)$ variates.

*Proof.* By induction on $i \in \{1, \ldots, n\}$. Write
$\mathrm{last}_i$ for the internal `last` after the $i$-th yield,
with $\mathrm{last}_0 = 0$.

*Base case* ($i = 1$). $\mathrm{remaining} = n$, $\mathrm{last} = 0$.
Compute $\mathrm{spacing} = \mathtt{first\_uniform}(rng, n)$,
distributed as $\mathrm{Beta}(1, n)$ by construction; by Lemma 2
this is the distribution of the minimum of $n$ iid
$\mathrm{Uniform}(0, 1)$ — i.e. of $U_{(1)}$. Yield is
$0 + 1 \cdot \mathrm{spacing} = \mathrm{spacing}$, so
$\mathrm{last}_1 \sim U_{(1)}$. ✓

*Inductive step.* Assume $(\mathrm{last}_1, \ldots, \mathrm{last}_i)$
has the joint distribution of $(U_{(1)}, \ldots, U_{(i)})$. Now
$\mathrm{remaining} = n - i$,
$\mathrm{spacing} = \mathtt{first\_uniform}(rng, n - i) \sim
\mathrm{Beta}(1, n - i)$. By Lemma 1, conditional on
$\mathrm{last}_i = u$, $U_{(i+1)}$ is the minimum of $n - i$ iid
$\mathrm{Uniform}(u, 1)$ draws, equal in distribution to
$u + (1 - u) \cdot \min(W_1, \ldots, W_{n-i})$ for $W_j$ iid
$\mathrm{Uniform}(0, 1)$. By Lemma 2 the inner min is
$\mathrm{Beta}(1, n - i)$, exactly the distribution of
$\mathrm{spacing}$. So
$\mathrm{last}_{i+1} = \mathrm{last}_i + (1 - \mathrm{last}_i)
\cdot \mathrm{spacing}$ has the correct conditional distribution
given $\mathrm{last}_i$, extending the hypothesis to step $i + 1$. ∎

#### Theorem 2 (correctness of `sample_indices`)

Let $w_1, \ldots, w_m \ge 0$ with $T = \sum_j w_j > 0$. Define iid
multinomial draws $K_1, \ldots, K_n$ with $\Pr[K_a = j] = w_j / T$.
Then the output sequence $J_1 \le \cdots \le J_n$ produced by
`sample_indices(rng, weights, n)` has the same joint distribution
as the sorted multinomial sample $(K_{(1)}, \ldots, K_{(n)})$.
(1-indexed in this proof; the code is 0-indexed.)

*Proof.* Let $W_j = w_1 + \cdots + w_j$, $F(j) = W_j / T$. The
inverse-CDF multinomial sampler draws $U_a \sim
\mathrm{Uniform}(0, 1)$ and sets

$$
K_a = \min \{\, j : F(j) > U_a \,\} = \min \{\, j : T \cdot U_a < W_j \,\}. \quad (\ast)
$$

This is correct because $F(j - 1) \le U_a < F(j)$ happens with
probability $w_j / T$. The map $U_a \mapsto K_a$ is monotone non-
decreasing, so sorting the $U_a$ and applying $(\ast)$ yields the
sorted multinomial sample:

```
(K₍₁₎, ..., K₍ₙ₎) = (φ(U₍₁₎), ..., φ(U₍ₙ₎))                  (∗∗)
```

where $\varphi(u) = \min \{\, j : T \cdot u < W_j \,\}$. By
Theorem 1, `sample_indices` yields $(U_{(1)}, \ldots, U_{(n)})$
distributed as the order statistics of $n$ iid
$\mathrm{Uniform}(0, 1)$. For each yielded $U_{(i)}$, the merge sets
$\mathrm{target} = T \cdot U_{(i)}$ and advances $j$ until
$\mathrm{target} \le \mathrm{cumulative} = W_j$, giving

$$
J_i = \min \{\, j : W_j \ge T \cdot U_{(i)} \,\}. \quad (\ast{\ast}\ast)
$$

Predicates $(\ast)$ and $(\ast{\ast}\ast)$ differ only on the event
$T \cdot U = W_j$, a measure-zero event under the continuous
uniform distribution. So $J_i = \varphi(U_{(i)})$ almost surely,
matching $(\ast\ast)$. ∎

### 5.2. Lemma 3 (floating-point boundary)

Suppose `weights[i]` are finite and non-negative with positive sum.
Then `sample_indices` cannot index past `weights.len() − 1`
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

(Tests written by Claude under direction; the threshold algebra
below has not been independently audited. Corrections welcome.)

#### 5.4.1. Aggregate budget and per-test allocation

The integration tests in `tests/statistical.rs` (run via `cargo
test`) check ten properties of `first_uniform`, `SortedUniforms`,
and the two samplers. All RNG seeds are fixed (`SmallRng`,
seed-based) so each test is a deterministic function of the
`rand` / `rand_distr` algorithm version pinned by `Cargo.lock`.

We aim for an **aggregate random-failure probability under correct
code below $10^{-9}$**. With ten tests this leaves $\sim 10^{-10}$
per test (Bonferroni union bound over tests). Each test internally
runs $S$ sub-checks (one per parameter setting); we Bonferroni-
correct again to give each sub-check a budget of $10^{-10}/S
\approx 10^{-11}$. The thresholds defined as constants at the top
of the test file (`KS_CRITICAL = 3.7`, `NORMAL_Z_SINGLE = 7.0`,
`NORMAL_Z_MAX_OVER_POS = 7.5`, `CHISQ_Z = 7.0`) are calibrated to
that per-sub-check budget; the per-test entries below show the
arithmetic.

#### 5.4.2. Reference formulae and citations

- **Two-sided Kolmogorov–Smirnov tail** (one- and two-sample):
  $\Pr[D > c/\sqrt{n_{\mathrm{eff}}}] \approx 2 \exp(-2c^2)$
  asymptotically. For two-sample with sizes $n_a, n_b$,
  $n_{\mathrm{eff}} = n_a n_b / (n_a + n_b)$. Standard derivation
  due to Smirnov (1939); a modern treatment is in Marsaglia, Tsang
  & Wang, *J. Stat. Softw.* 8(18), 2003 (also notes that the
  asymptotic underestimates the tail at small $n$, which we
  compensate for by oversizing the sample counts).
- **Wilson–Hilferty cube-root normalization of $\chi^2$**: if
  $X \sim \chi^2_d$, then
  $\bigl(X/d\bigr)^{1/3} \approx \mathcal{N}\!\bigl(1 -
  \tfrac{2}{9d}, \tfrac{2}{9d}\bigr)$, so the upper-tail
  critical value at standard-normal quantile $z$ is
  $d \bigl(1 - \tfrac{2}{9d} + z\sqrt{\tfrac{2}{9d}}\bigr)^3$.
  Original: Wilson & Hilferty, *Proc. Natl. Acad. Sci. USA*
  17(12), 1931, 684–688. This is the formula in the test
  helper `chisq_critical(dof, z)`.
- **Beta(1, k) moments**: $E[X] = 1/(k+1)$,
  $\mathrm{Var}[X] = k / [(k+1)^2 (k+2)]$. Standard; see e.g.
  Johnson, Kotz & Balakrishnan, *Continuous Univariate
  Distributions*, 2nd ed., vol. 2, §25.
- **Uniform order-statistic moments**: for the $i$-th of $n$
  sorted uniforms, $E[U_{(i)}] = i/(n+1)$ and
  $\mathrm{Var}[U_{(i)}] = i (n - i + 1) / [(n+1)^2 (n+2)]$.
  Standard; David & Nagaraja (2003) §2.4.

#### 5.4.3. The ten tests

For each test we list: function name, null hypothesis, statistic
computed, asymptotic distribution under the null, threshold and
budget arithmetic, and number of sub-checks $S$.

**1. Range check** — `range`. Deterministic; no randomness budget.
Asserts every `first_uniform` sample lies in $[0, 1)$ across
$k \in \{1, 2, 3, 5, 10, 100, 1000\}$, 100,000 samples each. A
violation indicates a real bug, not a tail event. ($S = 0$
randomness sub-checks.)

**2. Empirical moments of `first_uniform`** — `moments_first_uniform`.
Null: `first_uniform(k)` is distributed as $\mathrm{Beta}(1, k)$.
For $k \in \{1, 2, 3, 5, 10, 50, 200\}$ (so $S = 7$), draw
$n = 10^6$ samples and compare empirical mean and variance against
the closed-form moments. Mean check: $z = |\bar{x} - \mu| /
\sigma_{\bar{x}}$ where $\sigma_{\bar{x}} = \sqrt{\mathrm{Var}/n}$;
threshold `NORMAL_Z_SINGLE = 7.0` gives a per-sub-check tail of
$\Pr[|Z| > 7] \approx 2.6 \times 10^{-12}$, so
$2 S \cdot 2.6 \times 10^{-12} \approx 4 \times 10^{-11}$ for the
14 mean+variance sub-checks together — within budget. Variance
check: relative tolerance 2%, which at $n = 10^6$ is $\approx 14\sigma$
(swamped by the mean check above; not budget-limiting).

**3. One-sample KS against analytic CDF** — `ks_against_theory`.
Null: `first_uniform(k)` has CDF $F_k(x) = 1 - (1 - x)^k$. For
$k \in \{1, 2, 3, 5, 10, 50, 200\}$ ($S = 7$), draw $n = 50{,}000$
samples and compute $D = \sup_x |F_n(x) - F_k(x)|$. Threshold:
$D < c / \sqrt{n}$ with `KS_CRITICAL = 3.7`. Per-sub-check tail
$\approx 2 e^{-2 \cdot 3.7^2} \approx 2.5 \times 10^{-12}$;
total over $S$ sub-checks $\approx 1.8 \times 10^{-11}$ — within
budget.

**4. Two-sample KS vs. independent oracle** — `ks_against_min_oracle`.
Null: `first_uniform(k)` and the trivial `min(rng.gen::<f32>(),
...)` over $k$ draws come from the same distribution. Statistic
is the standard two-sample KS $D$; critical value $c \cdot
\sqrt{(n_a + n_b) / (n_a n_b)}$ with $c = 3.7$, $n_a = n_b =
20{,}000$, $S = 5$ values of $k$. Same per-sub-check tail as
test 3.

**5. Sorted-uniforms per-position moments** — `sorted_uniforms_moments`.
Null: position $i$ of `SortedUniforms::new(rng, n)` is distributed
as $U_{(i)}$ from $n$ iid uniforms. For $n \in \{5, 20, 100\}$
($S = 3$ outer, but $\sum n = 125$ inner per-position sub-checks,
each yielding mean+variance), draw 200,000 runs and compare
per-position empirical moments to the closed-form. Mean check
threshold `NORMAL_Z_MAX_OVER_POS = 7.5` gives
$\Pr[|Z| > 7.5] \approx 6.4 \times 10^{-14}$; multiplied by the
maximum 100 positions per outer setting yields
$\approx 6 \times 10^{-12}$ per outer — within budget. Variance
relative tolerance 5% (well into the no-fail regime at $n_{\mathrm{runs}}
= 2 \times 10^5$).

**6. Pooled sorted-uniforms KS** — `sorted_uniforms_pooled_ks`.
Null: pooling all positions of many `SortedUniforms` runs gives
samples from $\mathrm{Uniform}(0, 1)$ marginally. For $n \in
\{10, 100, 1000\}$ ($S = 3$), pool 50,000 total samples each and
KS-test against uniform CDF; $c = 3.7$ as in test 3.

**7. Sampler marginal $\chi^2$** — `sample_marginals_streaming`,
`sample_marginals_buffered`. Null: index $i$ appears with
probability $w_i / T$ in each sampler's output. For four weight
cases (`uniform-10`, `decreasing-8`, `peaky-12`,
`zeroes-mixed-6`), draw 4,000 runs of 50 samples each, accumulate
counts, and compute Pearson $\chi^2$ against expected counts
$n_{\mathrm{tot}} \cdot w_i / T$, dropping zero-weight cells.
Critical value via `chisq_critical(dof, CHISQ_Z = 7.0)`
(Wilson–Hilferty); per-sub-check tail
$\approx 1.3 \times 10^{-12}$. $S = 4$ weight cases × 2 samplers
= 8 sub-checks per pair, so each test contributes
$\approx 5 \times 10^{-12}$ — within budget. Zero-weight cells
are checked separately as exact equality (deterministic).

**8. Sampler vs. naive multinomial $\chi^2$** —
`sample_vs_multinomial_streaming`,
`sample_vs_multinomial_buffered`. Null: the sampler under test
and the naive reference (helper `naive_multinomial`, an O(m + n
log m) inverse-CDF construction) have the same per-index
probabilities. Two-sample Pearson $\chi^2$ on the
2-row contingency table of counts; same threshold as test 7.
$S = 3$ weight cases (`uniform-10`, `skewed-8`, `peaky-15`).

The reference `naive_multinomial` is implemented entirely in
**f64** internally — it builds the cumulative-weight array as
`Vec<f64>` and uses `partition_point` on a uniform-on-$[0, t)$
draw. No Kahan summation, no f32 prefix-sum noise. Its only
"trust assumption" is that `cum.partition_point(|&c| c <= target)`
implements binary search correctly (a `std` invariant). This is
intentionally distinct from the sampler under test so a bug in
either is exposed.

#### 5.4.4. Reproducibility caveat

The $10^{-9}$ budget is **per fixed `rand` / `rand_distr` algorithm
version and per fixed code usage of those crates**. If `rand`
bumps the algorithm of any of its samplers, or if our own pattern
of `rng.random()` / `Exp1.sample(rng)` calls changes, the seeded
sequences shift and a previously-passing test could land near a
threshold by coincidence. `Cargo.lock` is committed precisely to
pin those versions against this risk in CI and development. After
any change to crate code that touches RNG consumption (e.g. step
5 of the recent REVIEW pass: switching the streaming sampler to
an iterator), re-run `cargo test --release` and confirm no test
crosses its threshold; if one does, treat it as a real regression
rather than retuning the threshold.

### 5.5. Microbenchmark methodology

Benches live in `benches/bench.rs` and use [Divan](https://docs.rs/divan)
(per-iteration timing, outlier rejection, tabular summary;
`harness = false` in `Cargo.toml` so Divan supplies the test
runner). Three bench groups, each parameterized by either $k$ or
$m$:

**`first_uniform_per_call`** — one `first_uniform(rng, k)` call,
wrapped in `divan::black_box` so LLVM cannot fuse consecutive
iterations into a vectorized `powf` over a batch. The streaming
sampler's spacings recurrence invokes `first_uniform` one-at-a-
time (each call's output feeds the next call's `last`), so the
realistic deployment cost *is* scalar per-call. The fence pins
the measurement to that.

This is purely about *cross-call* vectorization. The libm internal
to a single `powf` call may itself use SIMD instructions to compute
that one result faster — `black_box` does not (and cannot) defeat
that, and we want the realistic per-call cost the libm delivers.
The ~10 ns/call measured on host x86 is exactly that.

**`pipeline_*_fenced`** — full sampler call with `black_box` only
at the outer API boundary. LLVM is free to optimize *inside* the
sampler. The buffered variant has two flat passes over `out`
(phase 1 Exp1 fill; phase 2 merge), and on hosts with SIMD those
passes do autovectorize — this is the realistic deployment number
on host x86. The streaming variant has the spacings-recurrence
data dependency and cannot vectorize internally, so its number
is the same scalar cost you'd see at deployment regardless.

**`pipeline_*_unfenced`** — same calls, no fences anywhere. LLVM
may additionally fuse work across iterations of the bench loop
(constant-fold, hoist, etc.). Together with the fenced variant
this brackets the realistic cost on host x86: the gap between
`pipeline_buffered_fenced` and `pipeline_buffered_unfenced` is
the upper bound on cross-call fusion benefit on this CPU. In
practice the gap is small (the per-call work is too large for
cross-iteration fusion to pay off) but having the bracket makes
the claim auditable rather than asserted.

(In short: the per-call bench fences cross-iteration vectorization
because that vectorization isn't available at deployment; the
fenced pipeline bench permits the internal vectorization that *is*
available; the unfenced pipeline bench bounds the additional gain
from inter-call fusion that almost certainly isn't.)

---

## 6. Performance

### 6.1. `first_uniform` per-call cost (host x86, fenced)

Median of Divan's per-iteration samples; `cargo bench`. Output
column `median` from the `first_uniform_per_call` group.

| k | median ns/call |
|----|----|
| 5    | 9.8 |
| 10   | 9.8 |
| 50   | 9.8 |
| 200  | 9.8 |
| 1000 | 9.8 |

Essentially flat in $k$. Dominated by the scalar `f32::powf` call;
the RNG (Xoshiro256++ in `SmallRng`) is sub-nanosecond per draw and
contributes ~1 ns at most. (The first bench in a Divan run can
show warmup-related jitter at the smallest $k$ — re-run if you
care about that cell. The other cells are stable across runs.)

### 6.2. Full sampling pipeline

`m = n`, weights `1..=m`, `f32`, SmallRng (Xoshiro256++)
seed-based. Median of Divan's per-iteration samples. ns/step is
ns/call divided by $(m + n) = 2m$.

**Fenced (API-boundary `black_box`, internal vectorization
allowed) vs. unfenced (nothing fenced, cross-iteration fusion
allowed):**

```
              streaming                       buffered
    m = n     fenced ns/step  unfenced       fenced ns/step  unfenced     C/B (fenced)
      100     14.97           15.72          11.72           10.97           1.28x
     1000     14.82           15.51          11.58           10.98           1.28x
    10000     14.55           15.20          11.41           10.95           1.28x
   100000     14.43           15.05          10.87           10.83           1.33x
  1000000     14.43           15.11          10.74           11.30           1.34x
```

Linear scaling across five orders of magnitude in both modes.
Per-step cost is dominated by the `Beta(1, k)` draw (streaming)
or the `Exp(1)` draw plus merge (buffered); cache effects don't
materially degrade large-$m$ performance because the weight array
is touched in a single forward sweep.

Fenced and unfenced numbers track each other within ~5% across
all cells — at this granularity (one sampler call is ~1–10 ms of
work for $m \ge 10^4$) cross-call fusion has nothing to amortize
across, so the fenced number *is* the realistic deployment cost
on this host. Internal autovectorization, by contrast, is real
and is what gives the buffered variant its ~1.3× edge.

### 6.3. Cortex-M4F expectations

On Cortex-M4F (no SIMD, scalar `powf` ~100–150 cycles, `Exp(1)`
Ziggurat ~30–60 cycles), the buffered variant is expected to win
by a larger margin than the ~1.32× we see on x86. We have no on-
target measurements yet.

---

## 7. Future directions

### 7.0. Note on splitting streaming/buffered into separate Cargo features

A natural-looking refactor would expose the streaming and buffered
samplers under separate Cargo features, on the theory that a
buffered-only `no_std` build could then drop `num-traits` (and
hence `libm`). It does not work: `rand_distr` 0.6 declares its
`num-traits` dependency with `default-features = false, features =
["libm"]` *unconditionally*, so as long as we use `Exp1` (or any
other `rand_distr` sampler) we transitively depend on `num-traits`
with the `libm` feature on. `cargo tree --no-default-features
--features libm` confirms this — `libm` v0.2 appears under
`num-traits` regardless of which `ltsis` feature we pick.

Conclusion: the refactor would change the public API surface
without removing any actual dependency. Not worth doing unless
either we drop `Exp1` (replace with our own Ziggurat) or
`rand_distr` changes its dependency-feature wiring.

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
