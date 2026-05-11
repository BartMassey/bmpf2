//! Sequential Importance Sampling (SIS) primitives — multinomial
//! sampling with replacement, in O(n) time.
//!
//! See `README.md` for a tutorial introduction and `INTERNALS.md` for
//! the algorithm specification, math proofs, and floating-point
//! correctness arguments.
//!
//! # API at a glance
//!
//! - [`resample_indices`] — the main entry point. Draws `out.len()`
//!   indices into `weights` iid with replacement, each with
//!   probability proportional to its weight. Output is in ascending
//!   order. Streaming: one `powf` call per output index.
//! - [`resample_indices_buffered`] — same signature and statistical
//!   contract, typically ~1.32× faster on x86 (more on hardware with
//!   a slow `powf`).
//! - [`SortedUniforms`] — iterator yielding `n` Uniform(0, 1) variates
//!   in ascending order in O(n) time. Useful in its own right outside
//!   resampling (e.g. inverse-CDF sampling where you want sorted
//!   output).
//! - [`first_uniform`] — low-level per-step primitive used by
//!   [`SortedUniforms`]. Samples the minimum of `k` iid Uniform(0, 1)
//!   draws (≡ `Beta(1, k)`). Most callers won't touch this directly.
//!
//! Indices are written as `u32`, not `usize`, so the API has the same
//! layout on every platform. Callers cast to `usize` to use them as
//! slice indices: `particles[out[i] as usize]`.
//!
//! # `no_std` support
//!
//! The library compiles in `no_std` mode and performs no allocation
//! — it operates over caller-supplied slices (`&[f32]` for weights,
//! `&mut [u32]` for output). Two mutually exclusive math-source
//! features:
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

mod first_uniform;
mod resample;
mod sorted_uniforms;

pub use first_uniform::first_uniform;
pub use resample::{resample_indices, resample_indices_buffered};
pub use sorted_uniforms::SortedUniforms;
