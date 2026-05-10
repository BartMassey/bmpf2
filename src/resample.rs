//! Linear-time perfect weighted resampling (Sequential Importance
//! Resampling with replacement, a.k.a. multinomial resampling).
//!
//! Given a vector of `m` weights (need not be normalized) and an
//! output count `n`, produces `n` indices drawn iid (with
//! replacement) with probability proportional to the weights. Output
//! is statistically equivalent to a multinomial draw of length `n`
//! on the weight distribution, in O(m + n) time.
//!
//! Two variants, identical signatures:
//!
//! - [`resample_indices`] — **streaming**. One `powf` per output
//!   index.
//! - [`resample_indices_buffered`] — **buffered**. Generates sorted
//!   uniforms via Gamma ratios (Exp(1) draws) instead of `powf`.
//!   Typically ~1.28× faster on x86; more on hardware with a slow
//!   `powf`. Repurposes `out` as scratch internally.
//!
//! Both produce the same statistical output (multinomial-distributed
//! indices in ascending order). Pick whichever fits your performance
//! budget; neither needs caller-supplied scratch.
//!
//! See `INTERNALS.md` for the algorithm specification, math proofs,
//! and floating-point correctness arguments.

use rand::Rng;
use rand_distr::{Distribution, Exp1};

use crate::sorted_uniforms::SortedUniforms;

// Kahan compensated add: `*sum += x` with running compensator `*c`.
// Used for every multi-term accumulator on the data path; reduces
// f32 prefix-sum error from O(n · 2⁻²⁴) to O(2⁻²⁴ · max|term|),
// effectively constant in n. See INTERNALS.md §4.6.
#[inline(always)]
fn kahan_add(sum: &mut f32, c: &mut f32, x: f32) {
    let y = x - *c;
    let t = *sum + y;
    *c = (t - *sum) - y;
    *sum = t;
}

/// Resample `out.len()` indices from `weights`, with each index
/// drawn iid (with replacement) with probability proportional to
/// its weight ("multinomial resampling"). Output is in ascending
/// order; permute `out` afterward if you need it shuffled.
///
/// Streaming variant: runs in O(`weights.len()` + `out.len()`)
/// time, allocates nothing, and uses one [`crate::first_uniform`]
/// call (one `powf`) per output index.
///
/// # Preconditions
/// - `weights` is nonempty.
/// - `weights.len()` ≤ `u32::MAX`.
/// - All entries of `weights` are finite and nonnegative.
/// - The sum of `weights` is strictly positive.
///
/// Violating any of these will panic in debug builds and produce
/// undefined (but memory-safe) output in release.
///
/// # See also
/// [`resample_indices_buffered`] — same signature and statistical
/// contract, typically ~1.28× faster on x86 (more on hardware with
/// a slow `powf`).
///
/// # Panics
/// Panics in debug if preconditions are violated. Always panics if
/// `weights.is_empty()`.
pub fn resample_indices<R: Rng + ?Sized>(rng: &mut R, weights: &[f32], out: &mut [u32]) {
    assert!(!weights.is_empty(), "weights must be nonempty");
    debug_assert!(
        weights.len() <= u32::MAX as usize,
        "weights.len() must fit in u32"
    );

    if out.is_empty() {
        return;
    }

    // Kahan-sum total weight in index order. The merge below re-walks
    // `weights` in the same index order with its own Kahan accumulator,
    // so by the time it consumes all weights its state matches `total`
    // bit-for-bit — load-bearing for Lemma 3 (INTERNALS.md §5.2).
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
    // weights[0] to (0, 0): that step yields (weights[0], 0). `j`
    // stays `usize` for slice indexing and is cast to `u32` on
    // store (lossless under the precondition above).
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
        *slot = j as u32;
    }
}

/// Buffered weighted resampler: same statistical contract and
/// signature as [`resample_indices`], typically ~1.28× faster on
/// x86 (more on hardware with a slow `powf`).
///
/// Generates sorted uniforms via the Gamma-ratio identity
/// (`U_(i) = (E_1 + ... + E_i) / (E_1 + ... + E_(n+1))` for `E_j`
/// iid Exp(1)) rather than via [`crate::first_uniform`], avoiding
/// the per-element `powf`. Internally repurposes `out` as scratch
/// (each `u32` slot temporarily holds the f32 bit pattern of an
/// Exp(1) draw via [`f32::to_bits`], later overwritten with the
/// output index).
///
/// See `INTERNALS.md` §4.4 for the algorithm and §5.2 for the
/// `target.min(total)` clip that keeps the merge bounded.
///
/// # Preconditions
/// - `weights` is nonempty.
/// - `weights.len()` ≤ `u32::MAX`.
/// - All entries of `weights` are finite and nonnegative.
/// - The sum of `weights` is strictly positive.
///
/// # Panics
/// Panics if `weights.is_empty()`. Panics in debug if weights are
/// non-finite/negative or sum to zero.
pub fn resample_indices_buffered<R: Rng + ?Sized>(rng: &mut R, weights: &[f32], out: &mut [u32]) {
    assert!(!weights.is_empty(), "weights must be nonempty");
    debug_assert!(
        weights.len() <= u32::MAX as usize,
        "weights.len() must fit in u32"
    );

    if out.is_empty() {
        return;
    }

    // Kahan-sum total weight; bit-for-bit reproducible against the
    // merge's incremental walk below (Lemma 3, INTERNALS.md §5.2).
    let mut total = 0.0_f32;
    let mut total_c = 0.0_f32;
    for &w in weights {
        debug_assert!(w.is_finite() && w >= 0.0, "weight must be finite and ≥ 0");
        kahan_add(&mut total, &mut total_c, w);
    }
    debug_assert!(total > 0.0, "total weight must be strictly positive");

    // Phase 1: fill `out` with E_1..E_n, encoded as f32 bit patterns
    // (each `u32` slot holds one Exp draw's bits exactly via
    // f32::to_bits). Kahan-accumulate G including E_{n+1}. Storing
    // E_{n+1} is unnecessary — we only need its contribution to G
    // so that S_n / G < 1 strictly.
    let mut g = 0.0_f32;
    let mut g_c = 0.0_f32;
    for slot in out.iter_mut() {
        let e: f32 = Exp1.sample(rng);
        *slot = e.to_bits();
        kahan_add(&mut g, &mut g_c, e);
    }
    let e_extra: f32 = Exp1.sample(rng);
    kahan_add(&mut g, &mut g_c, e_extra);

    // Phase 2: walk `out` left-to-right. Each slot is read back as
    // f32 (recovering the Exp draw stashed in Phase 1) then
    // overwritten with its output index in the same iteration.
    // cumulative_e accumulates S_i = E_1 + … + E_i (Kahan);
    // u_i = S_i / G is the i-th sorted uniform. Merge against
    // weights with Kahan on cumulative_w; the `target.min(total)`
    // clip handles the rare f32 case where u_n rounds to 1.0
    // exactly (INTERNALS.md §5.2).
    let inv_g = 1.0 / g;
    let mut cumulative_e = 0.0_f32;
    let mut ce_c = 0.0_f32;
    let mut j: usize = 0;
    let mut cumulative_w = weights[0];
    let mut cumulative_w_c = 0.0_f32;

    for slot in out.iter_mut() {
        let e = f32::from_bits(*slot);
        kahan_add(&mut cumulative_e, &mut ce_c, e);

        let u = cumulative_e * inv_g;
        let target = (total * u).min(total);
        while target > cumulative_w {
            j += 1;
            kahan_add(&mut cumulative_w, &mut cumulative_w_c, weights[j]);
        }
        *slot = j as u32;
    }
}
