//! Linear-time perfect weighted resampling (Sequential Importance
//! Resampling with replacement, a.k.a. multinomial resampling).
//!
//! Given a vector of `m` weights (need not be normalized) and an
//! output count `n`, produce `n` indices drawn iid (with replacement)
//! with probability proportional to the weights. Output is
//! statistically equivalent to a multinomial draw of length `n` on
//! the weight distribution, but runs in O(m + n) time using the
//! merge-with-sorted-uniforms construction of Massey 2008.
//!
//! Two variants:
//!
//! - [`resample_indices`] — **streaming**. No extra memory beyond
//!   the output slice. Calls [`crate::first_uniform`] (one `powf`)
//!   per output index.
//! - [`resample_indices_buffered`] — **buffered**. Caller supplies
//!   an `n`-element scratch buffer; in exchange, the per-element
//!   cost drops because it generates sorted uniforms via Gamma
//!   ratios (Exp(1) draws) instead of `powf`. Typically ~1.3×
//!   faster on x86; more on hardware with a slow `powf`.
//!
//! Both produce the same statistical output (multinomial-distributed
//! indices in ascending order). Pick whichever fits your memory
//! budget.
//!
//! Numerical robustness: every multi-term accumulator on the data
//! path uses Kahan compensated summation, so the per-accumulator
//! error is `O(ε)` rather than `O(n · ε)`.

use rand::Rng;
use rand_distr::{Distribution, Exp1};

use crate::sorted_uniforms::SortedUniforms;

/// Kahan compensated add: `*sum += x` with running compensator `*c`.
#[inline(always)]
fn kahan_add(sum: &mut f32, c: &mut f32, x: f32) {
    let y = x - *c;
    let t = *sum + y;
    *c = (t - *sum) - y;
    *sum = t;
}

/// Resample `out.len()` indices from `weights`, with each index drawn
/// iid (i.e. with replacement) with probability proportional to its
/// weight. Output is statistically equivalent to a multinomial draw
/// of length `out.len()` on the weight distribution — i.e. this is
/// "multinomial resampling".
///
/// Streaming: runs in O(`weights.len()` + `out.len()`) time,
/// allocates nothing, and uses one [`crate::first_uniform`] call
/// (one `powf`) per output index. The resulting indices are in
/// ascending order; callers who need them shuffled should permute
/// `out` afterward.
///
/// # Preconditions
/// - `weights` is nonempty.
/// - All entries of `weights` are finite and nonnegative.
/// - The sum of `weights` is strictly positive.
///
/// Violating any of these will panic in debug builds and produce
/// undefined (but memory-safe) output in release.
///
/// # See also
/// [`resample_indices_buffered`] — same statistical contract,
/// typically ~1.3× faster on x86 (more on hardware with a slow
/// `powf`), but requires a caller-supplied scratch buffer of length
/// `out.len()`.
///
/// # Panics
/// Panics in debug if preconditions are violated. Always panics if
/// `weights.is_empty()`.
pub fn resample_indices<R: Rng + ?Sized>(rng: &mut R, weights: &[f32], out: &mut [usize]) {
    assert!(!weights.is_empty(), "weights must be nonempty");

    if out.is_empty() {
        return;
    }

    // Total weight, Kahan-summed in index order. The merge below
    // re-walks `weights` in the same index order with its own Kahan
    // accumulator, so by the time it has consumed all weights its
    // state matches `total` bit-for-bit. See README Lemma 3.
    let mut total = 0.0_f32;
    let mut total_c = 0.0_f32;
    for &w in weights {
        debug_assert!(w.is_finite() && w >= 0.0, "weight must be finite and ≥ 0");
        kahan_add(&mut total, &mut total_c, w);
    }
    debug_assert!(total > 0.0, "total weight must be strictly positive");

    let n = out.len() as u32;
    let mut sorted = SortedUniforms::new(rng, n);

    // Streaming merge. `j` advances monotonically through `weights`;
    // `(cumulative, cumulative_c)` is the Kahan state for
    // `w_0 + ... + w_j`. Initialized as if we'd just Kahan-added
    // weights[0] to (0, 0): that step yields (weights[0], 0).
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

/// Buffered weighted resampler: same statistical contract as
/// [`resample_indices`], typically ~1.3× faster on x86 (more on
/// hardware with a slow `powf`), but requires a caller-supplied
/// scratch buffer of length `out.len()`.
///
/// # Algorithm
/// Draw `n + 1` Exp(1) variates `E_1, …, E_{n+1}`. Let
/// `G = E_1 + … + E_{n+1}` and `S_i = E_1 + … + E_i`. Then
/// `U_(i) = S_i / G` for `i = 1, …, n` are exactly the order
/// statistics of `n` iid Uniform(0, 1) draws (a standard result on
/// Gamma / Beta distribution ratios). The function fills `scratch`
/// with the first `n` Exp draws, accumulates `G` (with the (n+1)-th
/// draw added but not stored), then walks `scratch` left-to-right
/// computing `S_i / G` and merging against the cumulative weight
/// vector. Note that this routine does *not* go through
/// [`crate::first_uniform`] or [`crate::SortedUniforms`].
///
/// Compared to [`resample_indices`]:
///
/// - **No `powf` per element.** [`resample_indices`] calls
///   [`crate::first_uniform`] once per output index; the scalar
///   `powf` dominates per-call cost. The buffered variant replaces
///   this with an Exp(1) draw (Ziggurat, no transcendental on the
///   fast path) plus a multiplication by `1/G`.
/// - **Needs scratch.** Caller supplies `&mut [f32]` of length
///   `out.len()`. On 32-bit MCU this costs `4n` extra bytes.
/// - **Same statistical output.** Both methods produce indices
///   distributed as iid multinomial draws on `weights`, in ascending
///   order.
///
/// # Numerical robustness
/// All four prefix sums (`total`, the merge's `cumulative_w`, `G`,
/// and the merge's `cumulative_e`) use Kahan compensated summation,
/// reducing the f32 accumulator error from `O(n · 2⁻²⁴)` to
/// `O(2⁻²⁴ · max|term|)` — effectively constant.
///
/// The merge target is computed as `(total * u).min(total)`. The
/// `.min` clips the rare floating-point case where
/// `u_n = S_n / G` rounds up to `1.0` (in `f32`, this happens when
/// `E_{n+1}/G` falls below `~2⁻²⁵`, which has probability `~3%` at
/// `n = 10⁶`). Combined with bit-for-bit equality of `total` and
/// the merge's final `cumulative_w` (both are deterministic Kahan
/// sums of the same weight sequence from `(0, 0)`), this guarantees
/// the merge loop terminates within `weights.len()` without an
/// explicit bounds check.
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
        debug_assert!(w.is_finite() && w >= 0.0, "weight must be finite and ≥ 0");
        kahan_add(&mut total, &mut total_c, w);
    }
    debug_assert!(total > 0.0, "total weight must be strictly positive");

    // Phase 1: fill scratch with E_1..E_n, Kahan-accumulate G
    // including E_{n+1}. Storing E_{n+1} is unnecessary — we only
    // need its contribution to G so that S_n / G < 1 strictly.
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
    // uniform. Merge against weights with Kahan on `cumulative_w`
    // and the safeguard `target.min(total)`; see the doc comment
    // above.
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
