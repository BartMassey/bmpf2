//! Sample the minimum of `k` iid Uniform(0, 1) draws — i.e. the
//! distribution of the *first* (smallest) of `k` uniform variates.
//! Equivalently, `Beta(1, k)` in the standard parametrization.
//!
//! Two implementations are provided, exposed unconditionally so callers
//! can compare or force a specific one:
//!
//! - [`first_uniform_pow`]: direct inverse-CDF, `1 − u^(1/k)`.
//! - [`first_uniform_rejection`]: Exp(1)-proposal rejection sampler.
//!
//! The dispatcher [`first_uniform`] selects between them based on the
//! active feature (`pow` wins if both are enabled). All three are
//! `pub`.

// In no_std mode, `f32::ln` / `f32::powf` are not inherent methods; we
// reach them through the `num_traits::Float` trait, which under the
// `libm` feature dispatches to the `libm` crate. In std mode the
// inherent methods are used directly.
#[cfg(not(feature = "std"))]
use num_traits::Float;

use rand::Rng;
use rand_distr::{Distribution, Exp1};

/// Draw a sample distributed as the minimum of `k` iid Uniform(0, 1)
/// variates (equivalently, `Beta(1, k)` in standard notation).
///
/// This is the per-step primitive driving the order-statistic
/// recurrence in [`crate::SortedUniforms`].
///
/// With the default `pow` feature, this calls [`first_uniform_pow`].
/// With the `rejection` feature (and `pow` disabled), this calls
/// [`first_uniform_rejection`]. If both are enabled, `pow` wins.
///
/// # Panics
/// Panics if `k == 0`.
#[cfg(feature = "pow")]
#[inline]
pub fn first_uniform<R: Rng + ?Sized>(rng: &mut R, k: u32) -> f32 {
    first_uniform_pow(rng, k)
}

#[cfg(all(feature = "rejection", not(feature = "pow")))]
#[inline]
pub fn first_uniform<R: Rng + ?Sized>(rng: &mut R, k: u32) -> f32 {
    first_uniform_rejection(rng, k)
}

/// Direct inverse-CDF sampler for the min of `k` iid Uniform(0, 1):
/// returns `1 − u^(1/k)` for `u ~ Uniform(0, 1)`.
///
/// Always available, regardless of feature flags, so it can be used
/// directly when a caller wants to force the pow backend or compare
/// against [`first_uniform_rejection`].
///
/// To eliminate the (rare) edge case where `u` is sampled as exactly
/// 0 — which would yield `1 − 0^(1/k) = 1` and freeze the
/// order-statistic recurrence at `last = 1` — this function redraws
/// if `u == 0`. The probability of a redraw is `~2⁻²³` per call.
///
/// # Panics
/// Panics if `k == 0`.
pub fn first_uniform_pow<R: Rng + ?Sized>(rng: &mut R, k: u32) -> f32 {
    assert!(k >= 1, "k must be at least 1");
    let u: f32 = loop {
        let candidate: f32 = rng.gen();
        if candidate != 0.0 {
            break candidate;
        }
    };
    if k == 1 {
        // Beta(1, 1) is just Uniform(0, 1); 1 − u is also Uniform(0, 1),
        // but returning u keeps one f32 op fewer.
        u
    } else {
        1.0 - u.powf(1.0 / k as f32)
    }
}

/// Exp(1)-proposal rejection sampler for the min of `k` iid
/// Uniform(0, 1): proposes `Y ~ Exp(1)`, accepts in log space, and
/// returns `Y/k` on acceptance.
///
/// On hardware where `pow` is significantly more expensive than `log`,
/// this can be faster than [`first_uniform_pow`]. On hardware with a
/// fast vectorizable `pow` (most modern x86 libms) it is slower.
///
/// Costs per attempt: one Exp(1) draw, one Uniform draw, one `log`
/// call, a handful of multiplies. Acceptance rate is high for moderate
/// k (≈ 1/M_k, which approaches 1 as k → ∞; for k = 2,
/// `M_2 = e/2`, so acceptance is ≈ 0.74 per attempt).
///
/// Always available, regardless of feature flags.
///
/// # Panics
/// Panics if `k == 0`.
pub fn first_uniform_rejection<R: Rng + ?Sized>(rng: &mut R, k: u32) -> f32 {
    assert!(k >= 1, "k must be at least 1");

    if k == 1 {
        return rng.gen();
    }

    let kf = k as f32;
    let km1 = (k - 1) as f32;

    // log M_k = (k-1)·log(1 - 1/k) + 1, the log of the supremum of
    // the un-normalized acceptance ratio (1 - y/k)^(k-1)·e^y over
    // y ∈ [0, k]. Computed once per call. Numerically:
    // M_k → e^(1/k) as k → ∞, so log_m_k → 1/k for large k.
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
            return y / kf;
        }
    }
}
