//! Sampling from Beta(k, 1) — equivalently, the maximum of k i.i.d.
//! Uniform(0, 1) variates — for use in the linear-time perfect weighted
//! resampling algorithm of Massey (ICASSP 2008).
//!
//! All public APIs are `f32`. The realistic deployment target is the
//! Cortex-M4F (and similar single-precision FPUs), so the library
//! commits to single precision throughout. Where the obvious f32
//! algorithm would lose too much precision (the prefix-sum walks in
//! [`resample_indices`] and [`resample_indices_buffered`]), the
//! library uses Kahan compensated summation to recover O(ε) error
//! while staying entirely on the f32 FPU.
//!
//! Two implementations are provided. The default (`pow` feature) computes
//! `u.powf(1.0 / k)` directly. The alternative (`rejection` feature) uses
//! Exp(1)-proposed rejection sampling with a log-space acceptance test;
//! it is provided for hardware where the libm `pow` is significantly
//! slower than `log` plus an Exp(1) Ziggurat draw. With per-call
//! `black_box` fences the `pow` path is roughly 2–3× faster on modern
//! x86; without fences SIMD vectorization can widen this to ~10×. On
//! embedded targets the gap narrows and may invert. Profile on the
//! real target before choosing.
//!
//! Both implementations are exposed unconditionally as `beta_k_1_pow`
//! and `beta_k_1_rejection`, regardless of feature flags, so they can
//! be compared and tested together. The top-level `beta_k_1` symbol
//! dispatches to whichever one the active feature selects.
//!
//! ## `no_std` support
//!
//! The library compiles in `no_std` mode. The crate has two mutually
//! exclusive math-source features:
//!
//! - `std` (default): use the standard library's libm bindings via
//!   the inherent `f32::ln` / `f32::powf` methods.
//! - `libm`: use the [`libm`] crate via [`num_traits`] for `ln` and
//!   `powf`. Suitable for bare-metal targets.
//!
//! Enable exactly one. The library performs no allocation: it operates
//! over caller-supplied slices (`&[f32]` for weights, `&mut [usize]`
//! for resample output, `&mut [f32]` for scratch) and never calls into
//! `alloc`.
//!
//! [`libm`]: https://crates.io/crates/libm
//! [`num_traits`]: https://crates.io/crates/num-traits
//!
//! ## Precision
//!
//! Per-sample precision is limited by the 24-bit mantissa: each
//! `Beta(k, 1)` draw is accurate to `~6·10⁻⁸` in absolute terms. The
//! prefix-sum walks in [`resample_indices`] and
//! [`resample_indices_buffered`] would naively accrue `O(n · 2⁻²⁴)`
//! relative error, which becomes unusable around n ≈ 10⁵; both walks
//! therefore use Kahan compensated summation, reducing the bound to
//! `O(2⁻²⁴ · max|w|)`, effectively constant. The `pow` backend
//! produces exactly `u = 0` from the underlying RNG with probability
//! `~2⁻²³` per call; this is guarded by an internal redraw in
//! [`beta_k_1_pow`].

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!(
    "Enable exactly one of the `std` or `libm` features so transcendental \
     math (`ln`, `powf`) is available."
);

// In no_std mode, `f32::ln` / `f32::powf` are not inherent methods; we
// reach them through the `num_traits::Float` trait, which under the
// `libm` feature dispatches to the `libm` crate. In std mode the
// inherent methods are used directly.
#[cfg(not(feature = "std"))]
use num_traits::Float;

use rand::Rng;
use rand_distr::{Distribution, Exp1};

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
pub fn beta_k_1<R: Rng + ?Sized>(rng: &mut R, k: u32) -> f32 {
    beta_k_1_pow(rng, k)
}

#[cfg(all(feature = "rejection", not(feature = "pow")))]
#[inline]
pub fn beta_k_1<R: Rng + ?Sized>(rng: &mut R, k: u32) -> f32 {
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
/// redraw is `~2⁻²³` per call.
///
/// # Panics
/// Panics if `k == 0`.
pub fn beta_k_1_pow<R: Rng + ?Sized>(rng: &mut R, k: u32) -> f32 {
    assert!(k >= 1, "k must be at least 1");
    let u: f32 = loop {
        let candidate: f32 = rng.gen();
        if candidate != 0.0 {
            break candidate;
        }
    };
    if k == 1 {
        u
    } else {
        u.powf(1.0 / k as f32)
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
pub fn beta_k_1_rejection<R: Rng + ?Sized>(rng: &mut R, k: u32) -> f32 {
    assert!(k >= 1, "k must be at least 1");

    if k == 1 {
        return rng.gen();
    }

    let kf = k as f32;
    let km1 = (k - 1) as f32;

    // log M_k = (k-1)·log(1 - 1/k) + 1, the log of the supremum of the
    // un-normalized acceptance ratio (1 - y/k)^(k-1)·e^y over y ∈ [0,k].
    // Computed once per call. Numerically: M_k → e^(1/k) as k → ∞, so
    // log_m_k → 1/k for large k.
    let log_m_k = km1 * (1.0 - 1.0 / kf).ln() + 1.0;

    loop {
        let y: f32 = Exp1.sample(rng);
        if y >= kf {
            // Outside target support; reject and redraw. Probability ≈ e^(-k).
            continue;
        }

        let v: f32 = rng.gen();
        // log A_k(y) = (k-1)·log(1 - y/k) + y - log M_k, ≤ 0 by construction.
        let log_accept = km1 * (1.0 - y / kf).ln() + y - log_m_k;
        if v.ln() < log_accept {
            return 1.0 - y / kf;
        }
    }
}

// ---------------------------------------------------------------------------
// Test utilities, kept in the library so external tests can use them.
// ---------------------------------------------------------------------------

/// Independent oracle: draw k uniforms and return their max.
/// Useful as a non-`pow`, non-rejection reference distribution in tests.
pub fn beta_k_1_max_of_uniforms<R: Rng + ?Sized>(rng: &mut R, k: u32) -> f32 {
    let mut m = 0.0_f32;
    for _ in 0..k {
        let u: f32 = rng.gen();
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
///
/// Computed in `f64` because this is a sanity check on the *math*: the
/// constant `M_k` is an analytic choice independent of the library's
/// f32 commitment, and we want a tight check, not a check sloppy
/// enough to absorb f32 evaluation error.
pub fn verify_acceptance_bound(k: u32, n_grid: usize) -> f64 {
    if k == 1 {
        return 0.0;
    }
    let kf = k as f64;
    let km1 = (k - 1) as f64;
    let n_grid_f = n_grid as f64;
    let log_m_k = km1 * (1.0 - 1.0 / kf).ln() + 1.0;

    let mut max_log_accept: f64 = f64::NEG_INFINITY;
    for i in 1..n_grid {
        let y = kf * (i as f64) / n_grid_f;
        let log_accept = km1 * (1.0 - y / kf).ln() + y - log_m_k;
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
// Numerical robustness: the total weight is Kahan-summed in index order,
// and the merge's `cumulative` Kahan-sums weights in the same index order.
// Kahan summation is deterministic in its input sequence and starting
// state, so by the time the merge has consumed `weights[0..n]`,
// `cumulative` equals `total` bit-for-bit. Combined with `target = total
// · u_i < total = cumulative` (in exact arithmetic, since u_i < 1), this
// prevents the merge from running off the end of the weights array. See
// Lemma 3 in the README.

/// Kahan compensated add: `*sum += x` with running compensator `*c`.
#[inline(always)]
fn kahan_add(sum: &mut f32, c: &mut f32, x: f32) {
    let y = x - *c;
    let t = *sum + y;
    *c = (t - *sum) - y;
    *sum = t;
}

/// Streaming iterator yielding `n` uniform variates in ascending order.
///
/// The values are distributed as the order statistics of `n` i.i.d.
/// Uniform(0, 1) draws. Internally uses the spacings recurrence: each
/// `next()` call advances by `(1 - last) · spacing` where `spacing` is
/// `1 - Beta(k, 1)` with `k` counting down from the initial `n` to 1.
///
/// Holds a mutable reference to the RNG. Yields exactly `n` values, then
/// `None` thereafter.
pub struct SortedUniforms<'a, R: Rng + ?Sized> {
    rng: &'a mut R,
    remaining: u32,
    last: f32,
}

impl<'a, R: Rng + ?Sized> SortedUniforms<'a, R> {
    /// Create an iterator that will yield `n` sorted uniform variates.
    pub fn new(rng: &'a mut R, n: u32) -> Self {
        Self {
            rng,
            remaining: n,
            last: 0.0,
        }
    }
}

impl<'a, R: Rng + ?Sized> Iterator for SortedUniforms<'a, R> {
    type Item = f32;

    #[inline]
    fn next(&mut self) -> Option<f32> {
        if self.remaining == 0 {
            return None;
        }
        // spacing ~ Beta(1, k): the minimum of `k` uniforms, i.e. how far
        // the next sorted variate sits into the remaining range. Computed
        // here as `1 - Beta(k, 1)`. The cancellation loses a few low-order
        // bits, but downstream uses are insensitive at that level.
        let spacing = 1.0 - beta_k_1(self.rng, self.remaining);
        self.last = self.last + (1.0 - self.last) * spacing;
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
/// # See also
/// [`resample_indices_buffered`] — same statistical contract, faster
/// per element on hardware with a slow `pow`, but requires an extra
/// scratch buffer of length `n`.
///
/// # Panics
/// Panics in debug if preconditions are violated. Always panics if
/// `weights.is_empty()`.
pub fn resample_indices<R: Rng + ?Sized>(rng: &mut R, weights: &[f32], out: &mut [usize]) {
    assert!(!weights.is_empty(), "weights must be nonempty");

    if out.is_empty() {
        return;
    }

    // Total weight, Kahan-summed in index order. The merge below re-walks
    // `weights` in the same index order with its own Kahan accumulator,
    // so by the time it has consumed all weights its state matches `total`
    // bit-for-bit. See module-level comment / Lemma 3 in the README.
    let mut total = 0.0_f32;
    let mut total_c = 0.0_f32;
    for &w in weights {
        debug_assert!(
            w.is_finite() && w >= 0.0,
            "weight must be finite and ≥ 0"
        );
        kahan_add(&mut total, &mut total_c, w);
    }
    debug_assert!(total > 0.0, "total weight must be strictly positive");

    let n = out.len() as u32;
    let mut sorted = SortedUniforms::new(rng, n);

    // Streaming merge. `j` advances monotonically through `weights`;
    // `(cumulative, cumulative_c)` is the Kahan state for w_0 + ... + w_j.
    // Initialized as if we'd just Kahan-added weights[0] to (0, 0): that
    // step yields (weights[0], 0).
    let mut j: usize = 0;
    let mut cumulative = weights[0];
    let mut cumulative_c = 0.0_f32;

    for slot in out.iter_mut() {
        // SortedUniforms always yields exactly `n` values.
        let u = sorted.next().expect("SortedUniforms exhausted prematurely");
        let target = total * u;
        while target > cumulative {
            j += 1;
            kahan_add(&mut cumulative, &mut cumulative_c, weights[j]);
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
/// - **Needs scratch.** Caller supplies `&mut [f32]` of length
///   `out.len()`. On 32-bit MCU this costs `4n` extra bytes.
/// - **Same statistical contract.** Both methods produce indices
///   distributed as i.i.d. multinomial draws on `weights`, in
///   ascending order.
///
/// # Numerical robustness
/// All four prefix sums (`total`, the merge's `cumulative_w`, `G`,
/// and the merge's `cumulative_e`) use Kahan compensated summation,
/// reducing the f32 accumulator error from `O(n · 2⁻²⁴)` to
/// `O(2⁻²⁴ · max|term|)` — effectively constant.
///
/// The merge target is computed as `(total * u).min(total)`. The
/// `.min` clips the rare floating-point case where `u_n = S_n / G`
/// rounds up to `1.0` (in `f32`, this happens when `E_{n+1}/G` falls
/// below `~2⁻²⁵`, which has probability `~3%` at `n = 10⁶`).
/// Combined with bit-for-bit equality of `total` and the merge's
/// final `cumulative_w` (both are deterministic Kahan sums of the
/// same weight sequence from `(0, 0)`), this guarantees the merge
/// loop terminates within `weights.len()` without an explicit bounds
/// check.
///
/// # Preconditions
/// - `weights` is nonempty.
/// - `scratch.len() == out.len()`.
/// - All entries of `weights` are finite and nonnegative.
/// - The sum of `weights` is strictly positive.
///
/// # Panics
/// Panics if `weights.is_empty()` or if `scratch.len() != out.len()`.
/// Panics in debug if weights are non-finite/negative or sum to zero.
pub fn resample_indices_buffered<R: Rng + ?Sized>(
    rng: &mut R,
    weights: &[f32],
    out: &mut [usize],
    scratch: &mut [f32],
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

    // Kahan-sum total weight; bit-for-bit reproducible against the
    // merge's incremental walk below (Lemma 3).
    let mut total = 0.0_f32;
    let mut total_c = 0.0_f32;
    for &w in weights {
        debug_assert!(
            w.is_finite() && w >= 0.0,
            "weight must be finite and ≥ 0"
        );
        kahan_add(&mut total, &mut total_c, w);
    }
    debug_assert!(total > 0.0, "total weight must be strictly positive");

    // Phase 1: fill scratch with E_1..E_n, Kahan-accumulate G including
    // E_{n+1}. Storing E_{n+1} is unnecessary — we only need its
    // contribution to G so that S_n / G < 1 strictly.
    let mut g = 0.0_f32;
    let mut g_c = 0.0_f32;
    for slot in scratch.iter_mut() {
        let e: f32 = Exp1.sample(rng);
        *slot = e;
        kahan_add(&mut g, &mut g_c, e);
    }
    let e_extra: f32 = Exp1.sample(rng);
    kahan_add(&mut g, &mut g_c, e_extra);

    // Phase 2: walk scratch left-to-right. cumulative_e accumulates
    // S_i = E_1 + … + E_i (Kahan); u_i = S_i / G is the i-th sorted
    // uniform. Merge against weights with Kahan on `cumulative_w` and
    // the safeguard `target.min(total)`; see the doc comment above.
    let inv_g = 1.0 / g;
    let mut cumulative_e = 0.0_f32;
    let mut ce_c = 0.0_f32;
    let mut j: usize = 0;
    let mut cumulative_w = weights[0];
    let mut cumulative_w_c = 0.0_f32;

    for (slot_in, slot_out) in scratch.iter().zip(out.iter_mut()) {
        kahan_add(&mut cumulative_e, &mut ce_c, *slot_in);

        let u = cumulative_e * inv_g;
        let target = (total * u).min(total);
        while target > cumulative_w {
            j += 1;
            kahan_add(&mut cumulative_w, &mut cumulative_w_c, weights[j]);
        }
        *slot_out = j;
    }
}
