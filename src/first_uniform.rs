//! Sample the minimum of `k` iid Uniform(0, 1) draws — equivalently,
//! `Beta(1, k)`. Used internally by [`crate::SortedUniforms`].

// In no_std mode, `f32::powf` is not an inherent method; we reach it
// through the `num_traits::Float` trait, which under the `libm`
// feature dispatches to the `libm` crate. In std mode the inherent
// method is used directly.
#[cfg(not(feature = "std"))]
use num_traits::Float;

use rand::{Rng, RngExt};

/// Draw a sample distributed as the minimum of `k` iid Uniform(0, 1)
/// variates (equivalently, `Beta(1, k)` in standard notation).
///
/// The per-step primitive driving the order-statistic recurrence in
/// [`crate::SortedUniforms`]. Most callers won't need this directly.
///
/// # Panics
/// Panics if `k == 0`.
pub fn first_uniform<R: Rng + ?Sized>(rng: &mut R, k: u32) -> f32 {
    assert!(k >= 1, "k must be at least 1");
    let u: f32 = rng.random();
    if k == 1 {
        // Beta(1, 1) is Uniform(0, 1); rng.random already returns that.
        // Equivalent to `1 − (1 − u)^1 = u`.
        u
    } else {
        // Inverse CDF of Beta(1, k). The form `1 − (1 − u)^(1/k)`
        // (rather than the algebraically-equivalent `1 − u^(1/k)`) is
        // well-behaved across the entire f32 `rng.random` range — each
        // of the 2²⁴ input bins maps to a distinct output in [0, 1)
        // with no special-case redraw needed. See INTERNALS.md §4.1.
        1.0 - (1.0 - u).powf(1.0 / k as f32)
    }
}
