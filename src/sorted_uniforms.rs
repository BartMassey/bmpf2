//! Streaming iterator yielding `n` Uniform(0, 1) variates in
//! ascending order in O(n) time.

use core::iter::FusedIterator;

use rand::Rng;

use crate::first_uniform::first_uniform;

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
