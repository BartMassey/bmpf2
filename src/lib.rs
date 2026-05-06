//! Sampling from Beta(k, 1) — equivalently, the maximum of k i.i.d.
//! Uniform(0, 1) variates — for use in the linear-time perfect weighted
//! resampling algorithm of Massey (ICASSP 2008).
//!
//! Two implementations are provided. The default (`pow` feature) computes
//! `u.powf(1.0 / k)` directly. The alternative (`rejection` feature) uses
//! Exp(1)-proposed rejection sampling with a log-space acceptance test;
//! it is provided for hardware where the libm `pow` is significantly
//! slower than `log` plus an Exp(1) Ziggurat draw. On modern desktop
//! hardware the `pow` path is roughly 10× faster; on embedded targets
//! the gap narrows and may invert. Profile on the real target before
//! choosing.
//!
//! Both implementations are exposed unconditionally as `beta_k_1_pow`
//! and `beta_k_1_rejection`, regardless of feature flags, so they can
//! be compared and tested together. The top-level `beta_k_1` symbol
//! dispatches to whichever one the active feature selects.
//!
//! All public APIs are generic over the float type via the [`BetaFloat`]
//! trait, which is implemented for `f32` and `f64`. The `f32` path is
//! intended for memory-constrained MCU targets with a single-precision
//! FPU (e.g. Cortex-M4F); the `f64` path is the default for desktop and
//! for any case where 10⁶-particle precision matters.
//!
//! ## `no_std` support
//!
//! The library compiles in `no_std` mode. The crate has two mutually
//! exclusive math-source features:
//!
//! - `std` (default): use the standard library's libm bindings.
//! - `libm`: use the [`libm`] crate via [`num_traits`] for `ln` and
//!   `powf`. Suitable for bare-metal targets.
//!
//! Enable exactly one. The library performs no allocation: it operates
//! over caller-supplied slices (`&[F]` for weights, `&mut [usize]` for
//! resample output) and never calls into `alloc`.
//!
//! [`libm`]: https://crates.io/crates/libm
//! [`num_traits`]: https://crates.io/crates/num-traits
//!
//! ## Float type tradeoffs
//!
//! The choice between `f32` and `f64` has both performance and accuracy
//! consequences. Pick based on the largest `n` you expect to resample.
//!
//! ### `f64` — ~15–16 decimal digits
//!
//! The default. The sum accumulators inside [`resample_indices`] (the
//! `total = Σ weights` walk and the matching `cumulative` walk in the
//! merge) accrue relative error of order `n · 2⁻⁵²`, which is below
//! `10⁻⁹` even at `n = 10⁶`. Use `f64` whenever sample-count error
//! matters or when you have no reason not to.
//!
//! ### `f32` — ~7 decimal digits
//!
//! Faster and half the memory on hardware with a single-precision FPU.
//! Suitable for typical embedded particle filters (10²–10⁴ particles).
//! The relevant precision bounds:
//!
//! - **Per-sample precision** is limited by the 24-bit mantissa: each
//!   `Beta(k, 1)` draw is accurate to `~6·10⁻⁸` in absolute terms, with
//!   the usual caveat that values close to 1 lose representable
//!   resolution in their distance from 1. For the sorted-uniforms
//!   recurrence in [`SortedUniforms`] this is generally fine; later
//!   spacings are pre-multiplied by `(1 − last)` and shrink with
//!   `last → 1`, so the absolute step size matches the local
//!   resolution.
//!
//! - **`resample_indices` accumulator error** scales as
//!   `O(n · 2⁻²⁴)` because the implementation uses a naive running
//!   sum (no Kahan compensation) for both `total` and the merge's
//!   `cumulative`. Concretely, the relative error in the cumulative
//!   prefix sum is approximately:
//!
//!   | `n`     | rel. error      |
//!   |---------|-----------------|
//!   | 10²     | ~6·10⁻⁶         |
//!   | 10³     | ~6·10⁻⁵         |
//!   | 10⁴     | ~6·10⁻⁴         |
//!   | 10⁵     | ~6·10⁻³         |
//!   | 10⁶     | ~6·10⁻² (unusable) |
//!
//!   Recommended: `f32` for `n ≤ 2¹⁶ ≈ 65 000`, `f64` beyond.
//!   The boundary is where per-sample selection bias from cumulative
//!   rounding becomes comparable to the natural Monte-Carlo standard
//!   error (`~1/√n`). For applications more sensitive to tail
//!   probabilities (rare events, high-weight peaks), tighten the
//!   bound by an order of magnitude.
//!
//! - **`pow`-backend zero edge case**: `f32` produces exactly
//!   `u = 0` from the underlying RNG with probability `~2⁻²³` per
//!   call, vs. `~2⁻⁵²` for `f64`. Either of these would yield
//!   `0^(1/k) = 0` and bias [`SortedUniforms`] (where `spacing = 1`
//!   would push `last` to its current value of `1` exactly). The
//!   library guards against this internally by redrawing on
//!   `u == 0` inside [`beta_k_1_pow`]; no caller action required.
//!
//! - **Rejection backend** is unaffected by the zero edge case: its
//!   output is `1 − y/k` for `y < k`, strictly less than 1 by
//!   construction, with no zero on the other end.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!(
    "Enable exactly one of the `std` or `libm` features so transcendental \
     math (`ln`, `powf`) is available."
);

use num_traits::Float;
use rand::Rng;
use rand_distr::{Distribution, Exp1};

// ---------------------------------------------------------------------------
// Float abstraction.
// ---------------------------------------------------------------------------

/// Floats supported by the sampler. Combines `num_traits::Float` (for the
/// arithmetic and transcendental ops) with the per-type RNG sampling
/// primitives we need.
///
/// Implemented for `f32` and `f64`. The `Exp(1)` and `Uniform(0, 1)`
/// distributions in `rand_distr` are not generic over the float type —
/// each type has its own `Distribution` impl using a per-type Ziggurat —
/// so we wrap those calls behind associated functions here.
pub trait BetaFloat: Float {
    /// Draw a single Exp(1) variate using `rand_distr`'s per-type Ziggurat.
    fn sample_exp1<R: Rng + ?Sized>(rng: &mut R) -> Self;
    /// Draw a single Uniform(0, 1) variate.
    fn sample_uniform<R: Rng + ?Sized>(rng: &mut R) -> Self;
}

impl BetaFloat for f32 {
    #[inline]
    fn sample_exp1<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Exp1.sample(rng)
    }
    #[inline]
    fn sample_uniform<R: Rng + ?Sized>(rng: &mut R) -> Self {
        rng.gen()
    }
}

impl BetaFloat for f64 {
    #[inline]
    fn sample_exp1<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Exp1.sample(rng)
    }
    #[inline]
    fn sample_uniform<R: Rng + ?Sized>(rng: &mut R) -> Self {
        rng.gen()
    }
}

// ---------------------------------------------------------------------------
// Top-level dispatch: chooses an implementation based on active features.
// ---------------------------------------------------------------------------

/// Draw X ~ Beta(k, 1).
///
/// With the default `pow` feature, this calls `beta_k_1_pow`.
/// With the `rejection` feature (and `pow` disabled), this calls
/// `beta_k_1_rejection`. If both are enabled, `pow` wins.
///
/// # Panics
/// Panics if `k == 0`.
#[cfg(feature = "pow")]
#[inline]
pub fn beta_k_1<F: BetaFloat, R: Rng + ?Sized>(rng: &mut R, k: u32) -> F {
    beta_k_1_pow(rng, k)
}

#[cfg(all(feature = "rejection", not(feature = "pow")))]
#[inline]
pub fn beta_k_1<F: BetaFloat, R: Rng + ?Sized>(rng: &mut R, k: u32) -> F {
    beta_k_1_rejection(rng, k)
}

#[cfg(not(any(feature = "pow", feature = "rejection")))]
compile_error!("At least one of the `pow` or `rejection` features must be enabled.");

// ---------------------------------------------------------------------------
// Implementation 1: direct pow.
// ---------------------------------------------------------------------------

/// Direct sampler: U ~ Uniform(0,1), return U^(1/k).
///
/// Always available, regardless of feature flags, so it can serve as the
/// reference oracle in tests.
///
/// To eliminate the (rare) edge case where `U` is sampled as exactly 0 —
/// which would yield `0^(1/k) = 0` and bias downstream sorted-uniforms
/// generation — this function redraws if `U == 0`. The probability of a
/// redraw is ~2⁻²³ for `f32` and ~2⁻⁵² for `f64` per call.
///
/// # Panics
/// Panics if `k == 0`.
pub fn beta_k_1_pow<F: BetaFloat, R: Rng + ?Sized>(rng: &mut R, k: u32) -> F {
    assert!(k >= 1, "k must be at least 1");
    let u: F = loop {
        let candidate = F::sample_uniform(rng);
        if candidate != F::zero() {
            break candidate;
        }
    };
    if k == 1 {
        u
    } else {
        u.powf(F::one() / F::from(k).unwrap())
    }
}

// ---------------------------------------------------------------------------
// Implementation 2: Exp(1) rejection.
// ---------------------------------------------------------------------------

/// Rejection sampler: Exp(1) proposal, log-space acceptance test.
///
/// On hardware where `pow` is significantly more expensive than `log`,
/// this can be faster than `beta_k_1_pow`. On hardware with a fast
/// vectorizable `pow` (most modern x86 libms) it is slower.
///
/// Costs per attempt: one Exp(1) draw, one Uniform draw, one `log` call,
/// a handful of multiplies. Acceptance rate is high for moderate k
/// (≈ 1/M_k, which approaches 1 as k → ∞; for k=2, M_2 = e/2, so
/// acceptance is ≈ 0.74 per attempt).
///
/// Always available, regardless of feature flags.
///
/// # Panics
/// Panics if `k == 0`.
pub fn beta_k_1_rejection<F: BetaFloat, R: Rng + ?Sized>(rng: &mut R, k: u32) -> F {
    assert!(k >= 1, "k must be at least 1");

    if k == 1 {
        return F::sample_uniform(rng);
    }

    let one = F::one();
    let kf = F::from(k).unwrap();
    let km1 = F::from(k - 1).unwrap();

    // log M_k = (k-1)·log(1 - 1/k) + 1, the log of the supremum of the
    // un-normalized acceptance ratio (1 - y/k)^(k-1)·e^y over y ∈ [0,k].
    // Computed once per call. Numerically: M_k → e^(1/k) as k → ∞, so
    // log_m_k → 1/k for large k.
    let log_m_k = km1 * (one - one / kf).ln() + one;

    loop {
        let y: F = F::sample_exp1(rng);
        if y >= kf {
            // Outside target support; reject and redraw. Probability ≈ e^(-k).
            continue;
        }

        let v: F = F::sample_uniform(rng);
        // log A_k(y) = (k-1)·log(1 - y/k) + y - log M_k, ≤ 0 by construction.
        let log_accept = km1 * (one - y / kf).ln() + y - log_m_k;
        if v.ln() < log_accept {
            return one - y / kf;
        }
    }
}

// ---------------------------------------------------------------------------
// Test utilities, kept in the library so external tests can use them.
// ---------------------------------------------------------------------------

/// Independent oracle: draw k uniforms and return their max.
/// Useful as a non-`pow`, non-rejection reference distribution in tests.
pub fn beta_k_1_max_of_uniforms<F: BetaFloat, R: Rng + ?Sized>(rng: &mut R, k: u32) -> F {
    let mut m = F::zero();
    for _ in 0..k {
        let u: F = F::sample_uniform(rng);
        if u > m {
            m = u;
        }
    }
    m
}

/// Verify that the supremum constant M_k used by the rejection sampler
/// is correct: the acceptance probability A_k(Y) must satisfy
/// log A_k(Y) ≤ 0 for all Y ∈ [0, k). Returns the maximum of log A_k(Y)
/// observed on a fine grid; should be ≤ 0 (modulo floating-point error).
pub fn verify_acceptance_bound<F: Float>(k: u32, n_grid: usize) -> F {
    if k == 1 {
        return F::zero();
    }
    let one = F::one();
    let kf = F::from(k).unwrap();
    let km1 = F::from(k - 1).unwrap();
    let n_grid_f = F::from(n_grid).unwrap();
    let log_m_k = km1 * (one - one / kf).ln() + one;

    let mut max_log_accept: F = F::neg_infinity();
    for i in 1..n_grid {
        let y = kf * F::from(i).unwrap() / n_grid_f;
        let log_accept = km1 * (one - y / kf).ln() + y - log_m_k;
        if log_accept > max_log_accept {
            max_log_accept = log_accept;
        }
    }
    max_log_accept
}

// ===========================================================================
// Linear-time perfect weighted resampling (Sequential Importance Resampling)
// ===========================================================================
//
// Given a vector of m weights (need not be normalized) and an output count n,
// produce n indices drawn i.i.d. with probability proportional to the weights.
// Statistically equivalent to independent multinomial draws, but runs in
// O(m + n) time using the merge-with-sorted-uniforms trick of Massey 2008.
//
// The two phases are:
//
//   1. Generate n uniform variates in sorted (ascending) order, in O(n) time.
//      This is done with the order-statistic recurrence: each spacing
//      (u_i - u_{i-1}) / (1 - u_{i-1}) is independently distributed as the
//      minimum of (n - i + 1) uniforms — i.e., 1 - Beta(k, 1) where
//      k = n - i + 1.
//
//   2. Merge the sorted variates against cumulative weights in a single
//      pass, in O(m + n) time.
//
// Total: O(m + n), with the variate-generation phase being inherently
// serial (loop-carried `last`) but the merge phase being a streaming scan
// that the compiler can vectorize.
//
// Numerical robustness: the total weight is computed by walking `weights`
// in index order with `total += w`, exactly matching the order in which
// the merge accumulates `cumulative`. This guarantees that when the merge
// reaches the last weight, `cumulative == total` bit-for-bit, so the
// strict-less inequality `target = total · u_i < total = cumulative`
// (which holds because `u_i < 1` strictly) prevents any walk past the end.

/// Streaming iterator yielding `n` uniform variates in ascending order.
///
/// The values are distributed as the order statistics of `n` i.i.d.
/// Uniform(0, 1) draws. Internally uses the spacings recurrence: each
/// `next()` call advances by `(1 - last) · spacing` where `spacing` is
/// `1 - Beta(k, 1)` with `k` counting down from the initial `n` to 1.
///
/// Holds a mutable reference to the RNG. Yields exactly `n` values, then
/// `None` thereafter.
pub struct SortedUniforms<'a, F: BetaFloat, R: Rng + ?Sized> {
    rng: &'a mut R,
    remaining: u32,
    last: F,
}

impl<'a, F: BetaFloat, R: Rng + ?Sized> SortedUniforms<'a, F, R> {
    /// Create an iterator that will yield `n` sorted uniform variates.
    pub fn new(rng: &'a mut R, n: u32) -> Self {
        Self {
            rng,
            remaining: n,
            last: F::zero(),
        }
    }
}

impl<'a, F: BetaFloat, R: Rng + ?Sized> Iterator for SortedUniforms<'a, F, R> {
    type Item = F;

    #[inline]
    fn next(&mut self) -> Option<F> {
        if self.remaining == 0 {
            return None;
        }
        // spacing ~ Beta(1, k): the minimum of `k` uniforms, i.e. how far
        // the next sorted variate sits into the remaining range. Computed
        // here as `1 - Beta(k, 1)`. The cancellation loses a few low-order
        // bits, but downstream uses are insensitive at that level.
        let spacing = F::one() - beta_k_1::<F, _>(self.rng, self.remaining);
        self.last = self.last + (F::one() - self.last) * spacing;
        self.remaining -= 1;
        Some(self.last)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.remaining as usize;
        (r, Some(r))
    }
}

/// Resample `out.len()` indices from `weights`, with each index drawn
/// independently with probability proportional to its weight.
///
/// Runs in O(`weights.len()` + `out.len()`). The resulting indices are in
/// ascending order (because the underlying sorted-uniforms generator is);
/// callers who need them shuffled should permute `out` afterward.
///
/// # Preconditions
/// - `weights` is nonempty.
/// - All entries of `weights` are finite and nonnegative.
/// - The sum of `weights` is strictly positive.
///
/// Violating any of these will panic in debug builds and produce undefined
/// (but memory-safe) output in release.
///
/// # Precision
/// With `F = f32`, the prefix-sum accumulator accrues relative error of
/// order `n · 2⁻²⁴` (≈ `6·10⁻⁵` at `n = 10³`, ≈ `6·10⁻³` at `n = 10⁵`).
/// Recommended `n ≤ 2¹⁶` for `f32`; use `f64` beyond. With `F = f64`
/// the same bound is `n · 2⁻⁵²`, negligible for any realistic `n`.
/// See the crate-level "Float type tradeoffs" section.
///
/// # See also
/// [`resample_indices_buffered`] — same statistical contract, faster
/// per element on hardware with a slow `pow`, but requires an extra
/// scratch buffer of length `n`.
///
/// # Panics
/// Panics in debug if preconditions are violated. Always panics if
/// `weights.is_empty()`.
pub fn resample_indices<F: BetaFloat, R: Rng + ?Sized>(
    rng: &mut R,
    weights: &[F],
    out: &mut [usize],
) {
    assert!(!weights.is_empty(), "weights must be nonempty");

    if out.is_empty() {
        return;
    }

    // Total weight, computed by walking `weights` in the same order the
    // merge will accumulate `cumulative`. This bit-for-bit equality is
    // load-bearing for the boundary argument — see module docs above.
    let mut total = F::zero();
    for &w in weights {
        debug_assert!(
            w.is_finite() && w >= F::zero(),
            "weight must be finite and ≥ 0"
        );
        total = total + w;
    }
    debug_assert!(total > F::zero(), "total weight must be strictly positive");

    let n = out.len() as u32;
    let mut sorted = SortedUniforms::<F, R>::new(rng, n);

    // Streaming merge. `j` advances monotonically through `weights`;
    // `cumulative` is the running prefix sum w_0 + ... + w_j.
    let mut j: usize = 0;
    let mut cumulative = weights[0];

    for slot in out.iter_mut() {
        // SortedUniforms always yields exactly `n` values.
        let u = sorted.next().expect("SortedUniforms exhausted prematurely");
        let target = total * u;
        while target > cumulative {
            j += 1;
            cumulative = cumulative + weights[j];
        }
        *slot = j;
    }
}

/// Buffered weighted resampler ("Method B"): same statistical contract
/// as [`resample_indices`], faster on hardware with a slow `pow`, but
/// requires a caller-supplied scratch buffer of length `out.len()`.
///
/// # Algorithm
/// Draw `n + 1` Exp(1) variates `E_1, …, E_{n+1}`. Let
/// `G = E_1 + … + E_{n+1}` and `S_i = E_1 + … + E_i`. Then
/// `U_(i) = S_i / G` for `i = 1, …, n` are exactly the order
/// statistics of `n` i.i.d. Uniform(0, 1) draws (a standard result on
/// Gamma / Beta distribution ratios). The algorithm fills `scratch`
/// with the first `n` Exp draws, accumulates `G` (with the (n+1)-th
/// draw added but not stored), then walks `scratch` left-to-right
/// computing `S_i / G` and merging against the cumulative weight
/// vector.
///
/// Compared to [`resample_indices`] (streaming "Method C"):
///
/// - **No `pow` per element.** Method C calls `beta_k_1` once per
///   output index; the `pow` backend dominates per-call cost. Method B
///   replaces this with an Exp(1) draw plus a multiplication by
///   `1/G` — typically several times faster, especially on hardware
///   without a fast vectorized `powf`.
/// - **Needs scratch.** Caller supplies `&mut [F]` of length
///   `out.len()`. On 32-bit MCU with `F = f32` and using a separate
///   output buffer, this costs `4n` extra bytes.
/// - **Same statistical contract.** Both methods produce indices
///   distributed as i.i.d. multinomial draws on `weights`, in
///   ascending order.
///
/// # Numerical robustness
/// Both `G` and the running prefix sum `cumulative_e` are accumulated
/// using Kahan summation. Without compensation, the `f32` accumulator
/// would accrue `O(n · 2⁻²⁴)` relative error — about 6% at `n = 10⁶`,
/// enough to perceptibly bias `U_(i)`. With Kahan, the error reduces
/// to `O(2⁻²⁴ · max|E_i|) ≈ O(ln n / 2²⁴)`, effectively constant.
/// `f64` doesn't need it for any realistic `n`, but the cost is
/// negligible (a few extra adds per element) so it is unconditional.
///
/// The merge target is computed as `(total * u).min(total)`. The
/// `.min` clips the rare floating-point case where `u_n = S_n / G`
/// rounds up to `1.0` (in `f32`, this happens when `E_{n+1}/G` falls
/// below `~2⁻²⁵`, which has probability `~3%` at `n = 10⁶`).
/// Combined with same-order accumulation of `total` and the merge's
/// `cumulative_w`, this guarantees the merge loop terminates within
/// `weights.len()` without an explicit bounds check.
///
/// # Preconditions
/// - `weights` is nonempty.
/// - `scratch.len() == out.len()`.
/// - All entries of `weights` are finite and nonnegative.
/// - The sum of `weights` is strictly positive.
///
/// # Precision
/// With `F = f32` the prefix-sum-of-weights accumulator accrues
/// `O(n · 2⁻²⁴)` relative error; recommended `n ≤ 2¹⁶`. Beyond that
/// switch to `F = f64`. The Kahan-protected `G` and `cumulative_e`
/// accumulators introduced by this routine are not the limiting
/// factor — the weight sum is.
///
/// # Panics
/// Panics if `weights.is_empty()` or if `scratch.len() != out.len()`.
/// Panics in debug if weights are non-finite/negative or sum to zero.
pub fn resample_indices_buffered<F: BetaFloat, R: Rng + ?Sized>(
    rng: &mut R,
    weights: &[F],
    out: &mut [usize],
    scratch: &mut [F],
) {
    assert!(!weights.is_empty(), "weights must be nonempty");
    assert_eq!(
        scratch.len(),
        out.len(),
        "scratch length must equal out length"
    );

    if out.is_empty() {
        return;
    }

    // Same-order accumulation invariant (see Lemma 3 in README): walk
    // `weights` here in index order, then again in the merge below, so
    // the final `cumulative_w` equals `total` bit-for-bit.
    let mut total = F::zero();
    for &w in weights {
        debug_assert!(
            w.is_finite() && w >= F::zero(),
            "weight must be finite and ≥ 0"
        );
        total = total + w;
    }
    debug_assert!(total > F::zero(), "total weight must be strictly positive");

    // Phase 1: fill scratch with E_1..E_n, Kahan-accumulate G including
    // E_{n+1}. Storing E_{n+1} is unnecessary — we only need its
    // contribution to G so that S_n / G < 1 strictly.
    let mut g = F::zero();
    let mut g_c = F::zero();
    for slot in scratch.iter_mut() {
        let e = F::sample_exp1(rng);
        *slot = e;
        let y = e - g_c;
        let t = g + y;
        g_c = (t - g) - y;
        g = t;
    }
    let e_extra = F::sample_exp1(rng);
    let y = e_extra - g_c;
    g = g + y; // last add; compensation no longer needed

    // Phase 2: walk scratch left-to-right. cumulative_e accumulates
    // S_i = E_1 + … + E_i (also Kahan); u_i = S_i / G is the i-th
    // sorted uniform. Merge against weights with the safeguard
    // `target.min(total)`; see the doc comment above.
    let inv_g = F::one() / g;
    let mut cumulative_e = F::zero();
    let mut ce_c = F::zero();
    let mut j: usize = 0;
    let mut cumulative_w = weights[0];

    for (slot_in, slot_out) in scratch.iter().zip(out.iter_mut()) {
        let y = *slot_in - ce_c;
        let t = cumulative_e + y;
        ce_c = (t - cumulative_e) - y;
        cumulative_e = t;

        let u = cumulative_e * inv_g;
        let target = (total * u).min(total);
        while target > cumulative_w {
            j += 1;
            cumulative_w = cumulative_w + weights[j];
        }
        *slot_out = j;
    }
}
