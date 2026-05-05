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
pub fn beta_k_1<R: Rng + ?Sized>(rng: &mut R, k: u32) -> f64 {
    beta_k_1_pow(rng, k)
}

#[cfg(all(feature = "rejection", not(feature = "pow")))]
#[inline]
pub fn beta_k_1<R: Rng + ?Sized>(rng: &mut R, k: u32) -> f64 {
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
/// # Panics
/// Panics if `k == 0`.
pub fn beta_k_1_pow<R: Rng + ?Sized>(rng: &mut R, k: u32) -> f64 {
    assert!(k >= 1, "k must be at least 1");
    let u: f64 = rng.gen();
    if k == 1 {
        u
    } else {
        u.powf(1.0 / k as f64)
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
pub fn beta_k_1_rejection<R: Rng + ?Sized>(rng: &mut R, k: u32) -> f64 {
    assert!(k >= 1, "k must be at least 1");

    if k == 1 {
        return rng.gen::<f64>();
    }

    let kf = k as f64;
    let km1 = (k - 1) as f64;

    // log M_k = (k-1)·log(1 - 1/k) + 1, the log of the supremum of the
    // un-normalized acceptance ratio (1 - y/k)^(k-1)·e^y over y ∈ [0,k].
    // Computed once per call. Numerically: M_k → e^(1/k) as k → ∞, so
    // log_m_k → 1/k for large k.
    let log_m_k = km1 * (1.0 - 1.0 / kf).ln() + 1.0;

    loop {
        let y: f64 = Exp1.sample(rng);
        if y >= kf {
            // Outside target support; reject and redraw. Probability ≈ e^(-k).
            continue;
        }

        let v: f64 = rng.gen();
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
pub fn beta_k_1_max_of_uniforms<R: Rng + ?Sized>(rng: &mut R, k: u32) -> f64 {
    let mut m = 0.0_f64;
    for _ in 0..k {
        let u: f64 = rng.gen();
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
pub fn verify_acceptance_bound(k: u32, n_grid: usize) -> f64 {
    if k == 1 {
        return 0.0;
    }
    let kf = k as f64;
    let km1 = (k - 1) as f64;
    let log_m_k = km1 * (1.0 - 1.0 / kf).ln() + 1.0;

    let mut max_log_accept: f64 = f64::NEG_INFINITY;
    for i in 1..n_grid {
        let y = kf * (i as f64) / (n_grid as f64);
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
pub struct SortedUniforms<'a, R: Rng + ?Sized> {
    rng: &'a mut R,
    remaining: u32,
    last: f64,
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
    type Item = f64;

    #[inline]
    fn next(&mut self) -> Option<f64> {
        if self.remaining == 0 {
            return None;
        }
        // spacing ~ Beta(1, k): the minimum of `k` uniforms, i.e. how far
        // the next sorted variate sits into the remaining range. Computed
        // here as `1 - Beta(k, 1)`. The cancellation loses a few low-order
        // bits, but downstream uses are insensitive at that level.
        let spacing = 1.0 - beta_k_1(self.rng, self.remaining);
        self.last += (1.0 - self.last) * spacing;
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
/// # Panics
/// Panics in debug if preconditions are violated. Always panics if
/// `weights.is_empty()`.
pub fn resample_indices<R: Rng + ?Sized>(rng: &mut R, weights: &[f64], out: &mut [usize]) {
    assert!(!weights.is_empty(), "weights must be nonempty");

    if out.is_empty() {
        return;
    }

    // Total weight, computed by walking `weights` in the same order the
    // merge will accumulate `cumulative`. This bit-for-bit equality is
    // load-bearing for the boundary argument — see module docs above.
    let mut total = 0.0_f64;
    for &w in weights {
        debug_assert!(w.is_finite() && w >= 0.0, "weight must be finite and ≥ 0");
        total += w;
    }
    debug_assert!(total > 0.0, "total weight must be strictly positive");

    let n = out.len() as u32;
    let mut sorted = SortedUniforms::new(rng, n);

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
            cumulative += weights[j];
        }
        *slot = j;
    }
}
