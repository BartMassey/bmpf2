//! Sequential Importance Sampling (SIS) primitives — multinomial
//! sampling with replacement, in O(n) time.
//!
//! See `README.md` for a tutorial introduction and `INTERNALS.md` for
//! the algorithm specification, math proofs, and floating-point
//! correctness arguments.
//!
//! # API at a glance
//!
//! - [`sample_indices`] — the main entry point. Returns an iterator
//!   yielding `n` indices into `weights` iid with replacement, each
//!   with probability proportional to its weight. Output is in
//!   ascending order. Streaming: one `powf` call per yielded index.
//! - [`sample_indices_buffered`] — buffered variant, typically
//!   ~1.32× faster on x86 (more on hardware with a slow `powf`).
//!   Takes an `&mut [u32]` buffer rather than returning an iterator
//!   (it uses the buffer as f32 scratch).
//! - [`SortedUniforms`] — iterator yielding `n` Uniform(0, 1) variates
//!   in ascending order in O(n) time. Useful in its own right outside
//!   sampling (e.g. inverse-CDF sampling where you want sorted
//!   output).
//! - [`first_uniform`] — low-level per-step primitive used by
//!   [`SortedUniforms`]. Samples the minimum of `k` iid Uniform(0, 1)
//!   draws (≡ `Beta(1, k)`). Most callers won't touch this directly.
//!
//! Indices are yielded/written as `u32`, not `usize`, so the API has
//! the same layout on every platform. Callers cast to `usize` at the
//! index site: `particles[j as usize]`.
//!
//! # `no_std` support
//!
//! The library compiles in `no_std` mode and performs no allocation.
//! [`sample_indices`] returns an iterator that the caller drives
//! (typically yielding into a caller-owned buffer); the buffered
//! variant operates entirely on a caller-supplied `&mut [u32]`.
//! Two mutually exclusive math-source features:
//!
//! - `std` (default): use std's inherent `f32::powf`.
//! - `libm`: use the [`libm`] crate via [`num_traits`] for `powf`.
//!   Suitable for bare-metal targets.
//!
//! Enable exactly one.
//!
//! [`libm`]: https://crates.io/crates/libm
//! [`num_traits`]: https://crates.io/crates/num-traits

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!(
    "Enable exactly one of the `std` or `libm` features so transcendental \
     math (`powf`) is available."
);

use core::iter::FusedIterator;

// In no_std mode, `f32::powf` is not an inherent method; we reach it
// through the `num_traits::Float` trait, which under the `libm`
// feature dispatches to the `libm` crate. In std mode the inherent
// method is used directly.
#[cfg(not(feature = "std"))]
use num_traits::Float;

use rand::{Rng, RngExt};
use rand_distr::{Distribution, Exp1};

// ---------------------------------------------------------------------------
// first_uniform
// ---------------------------------------------------------------------------

/// Draw a sample distributed as the minimum of `k` iid Uniform(0, 1)
/// variates (equivalently, `Beta(1, k)` in standard notation).
///
/// This is the per-step primitive driving the
/// order-statistic recurrence in [`SortedUniforms`]. Most callers
/// won't need this directly.
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

// ---------------------------------------------------------------------------
// SortedUniforms
// ---------------------------------------------------------------------------

/// Streaming iterator yielding `n` Uniform(0, 1) variates in ascending
/// order in O(n) time.
///
/// The yielded values are distributed exactly as the order statistics
/// of `n` iid Uniform(0, 1) draws — i.e. the same as drawing `n` iid
/// uniforms and sorting them, but produced one at a time without a
/// sort.
///
/// Holds a mutable reference to the RNG. Yields exactly `n` values,
/// then `None` thereafter.
///
/// See `INTERNALS.md` §3.1 / §5.1 for the spacings-recurrence
/// algorithm and its correctness proof.
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
        // Spacings recurrence: U_(i) = U_(i-1) + (1 - U_(i-1)) * Z,
        // where Z is the minimum of the `remaining` uniforms still
        // to come — distributed as Beta(1, remaining), supplied by
        // first_uniform.
        let spacing = first_uniform(self.rng, self.remaining);
        self.last = self.last + (1.0 - self.last) * spacing;
        self.remaining -= 1;
        Some(self.last)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.remaining as usize;
        (r, Some(r))
    }

    /// Returns the number of values still to be yielded, without
    /// consuming them. Overrides the default implementation, which
    /// would advance the RNG once per remaining value just to count.
    fn count(self) -> usize {
        self.remaining as usize
    }
}

/// `SortedUniforms` knows its exact remaining length (returned by
/// `size_hint`), so [`ExactSizeIterator::len`] is available.
impl<'a, R: Rng + ?Sized> ExactSizeIterator for SortedUniforms<'a, R> {}

/// Once `next()` has returned `None` (i.e. `remaining == 0`), all
/// subsequent calls also return `None` — the recurrence has no way
/// to restart.
impl<'a, R: Rng + ?Sized> FusedIterator for SortedUniforms<'a, R> {}

// ---------------------------------------------------------------------------
// sampling (streaming and buffered)
// ---------------------------------------------------------------------------

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

/// Build an iterator yielding `n` indices into `weights`, each
/// drawn iid (with replacement) with probability proportional to
/// its weight ("multinomial sampling"). Indices are yielded in
/// ascending order; collect into a `Vec` and shuffle afterward if
/// you need them in random order.
///
/// Streaming variant: runs in O(`weights.len()` + `n`) total,
/// allocates nothing, and uses one [`first_uniform`] call (one
/// `powf`) per yielded index.
///
/// # Preconditions
/// - `weights` is nonempty.
/// - `weights.len()` ≤ `u32::MAX`.
/// - All entries of `weights` are finite and nonnegative.
/// - The sum of `weights` is strictly positive.
///
/// All four preconditions are checked in release builds at
/// constructor time (the per-element finiteness check is fast
/// enough to leave on always — see INTERNALS.md §4.7).
///
/// # See also
/// [`sample_indices_buffered`] — buffered variant, typically
/// ~1.32× faster on x86 (more on hardware with a slow `powf`).
/// Takes an `&mut [u32]` buffer rather than returning an iterator
/// because it uses the buffer's slots as f32 scratch.
///
/// # Panics
/// Panics on any precondition violation. (Lazy iterator semantics
/// do *not* apply to validation: checks happen up-front so the
/// caller learns of bad inputs immediately.)
pub fn sample_indices<'a, R: Rng + ?Sized>(
    rng: &'a mut R,
    weights: &'a [f32],
    n: u32,
) -> SampleIndices<'a, R> {
    assert!(!weights.is_empty(), "weights must be nonempty");
    assert!(
        weights.len() <= u32::MAX as usize,
        "weights.len() must fit in u32"
    );

    // Kahan-sum total weight in index order. The merge re-walks
    // `weights` in the same index order with its own Kahan accumulator,
    // so by the time it consumes all weights its state matches `total`
    // bit-for-bit — load-bearing for Lemma 3 (INTERNALS.md §5.2).
    let mut total = 0.0_f32;
    let mut total_c = 0.0_f32;
    for &w in weights {
        assert!(w.is_finite() && w >= 0.0, "weight must be finite and ≥ 0");
        kahan_add(&mut total, &mut total_c, w);
    }
    assert!(total > 0.0, "total weight must be strictly positive");

    SampleIndices {
        sorted: SortedUniforms::new(rng, n),
        weights,
        total,
        // Streaming merge state. `j` advances monotonically through
        // `weights`; `(cumulative, cumulative_c)` is the Kahan state
        // for `w_0 + ... + w_j`. Initialized as if we'd just Kahan-
        // added weights[0] to (0, 0): that step yields (weights[0], 0).
        j: 0,
        cumulative: weights[0],
        cumulative_c: 0.0,
    }
}

/// Iterator returned by [`sample_indices`]. Yields `n` `u32`
/// indices into the original `weights` slice, in ascending order.
///
/// Implements [`ExactSizeIterator`] (the remaining count is exact)
/// and [`FusedIterator`] (returns `None` forever after exhaustion).
pub struct SampleIndices<'a, R: Rng + ?Sized> {
    sorted: SortedUniforms<'a, R>,
    weights: &'a [f32],
    total: f32,
    j: usize,
    cumulative: f32,
    cumulative_c: f32,
}

impl<'a, R: Rng + ?Sized> Iterator for SampleIndices<'a, R> {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<u32> {
        let u = self.sorted.next()?;
        let target = self.total * u;
        // `j` stays `usize` for slice indexing and is cast to `u32`
        // on yield (lossless under the length precondition).
        while target > self.cumulative {
            self.j += 1;
            kahan_add(
                &mut self.cumulative,
                &mut self.cumulative_c,
                self.weights[self.j],
            );
        }
        Some(self.j as u32)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.sorted.size_hint()
    }
}

impl<'a, R: Rng + ?Sized> ExactSizeIterator for SampleIndices<'a, R> {}
impl<'a, R: Rng + ?Sized> FusedIterator for SampleIndices<'a, R> {}

/// Buffered weighted sampler: same statistical contract and
/// signature as [`sample_indices`], typically ~1.32× faster on
/// x86 (more on hardware with a slow `powf`).
///
/// Generates sorted uniforms via the Gamma-ratio identity
/// (`U_(i) = (E_1 + ... + E_i) / (E_1 + ... + E_(n+1))` for `E_j`
/// iid Exp(1)) rather than via [`first_uniform`], avoiding the
/// per-element `powf`. Internally repurposes `out` as scratch
/// (each `u32` slot temporarily holds the f32 bit pattern of an
/// Exp(1) draw via [`f32::to_bits`], later overwritten with the
/// output index).
///
/// See `INTERNALS.md` §4.4 for the algorithm and §5.2 for the
/// `target.min(total)` clip that keeps the merge bounded.
///
/// # Preconditions
/// Same as [`sample_indices`]; all four are checked in release.
///
/// # Panics
/// Panics in release on any precondition violation.
pub fn sample_indices_buffered<R: Rng + ?Sized>(rng: &mut R, weights: &[f32], out: &mut [u32]) {
    assert!(!weights.is_empty(), "weights must be nonempty");
    assert!(
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
        assert!(w.is_finite() && w >= 0.0, "weight must be finite and ≥ 0");
        kahan_add(&mut total, &mut total_c, w);
    }
    assert!(total > 0.0, "total weight must be strictly positive");

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
