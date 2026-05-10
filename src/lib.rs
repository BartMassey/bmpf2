//! Sequential Importance Resampling (SIR) primitives, plus
//! (forthcoming) Bayesian Particle Filter (BPF) API on top.
//!
//! # What this crate does
//!
//! Given an array of `n` non-negative weights, [`resample_indices`]
//! draws `n` indices into the array, each chosen iid (i.e. **with
//! replacement**) with probability proportional to its weight, in
//! O(n) time. Statistically equivalent to drawing from a multinomial
//! distribution on the weights — this is the "multinomial
//! resampling" step used in Bayesian particle filters and other
//! sequential Monte Carlo methods. See `README.md` for a tutorial
//! introduction.
//!
//! Indices are written as `u32`, not `usize`, so the API has the
//! same layout on every platform (16-, 32-, or 64-bit). Callers
//! cast to `usize` to use them as slice indices.
//!
//! # API at a glance
//!
//! - [`resample_indices`] — the main entry point. **Streaming**:
//!   one `powf` call per output index.
//! - [`resample_indices_buffered`] — same signature and statistical
//!   contract, typically ~1.3× faster on x86 (more on hardware with
//!   a slow `powf`). Repurposes the `out` slice as scratch (each
//!   `usize` slot temporarily holds the f32 bit pattern of an Exp(1)
//!   draw via [`f32::to_bits`]); no extra allocation.
//! - [`SortedUniforms`] — the underlying order-statistic iterator
//!   used by [`resample_indices`]. Yields `n` Uniform(0, 1) variates
//!   in ascending order in O(n) time. Exposed because it's useful in
//!   its own right (e.g. for inverse-CDF sampling against any
//!   continuous distribution where you want sorted output).
//! - [`first_uniform`] — low-level per-step primitive used by
//!   [`SortedUniforms`]. Most callers won't touch this directly;
//!   samples the minimum of `k` iid Uniform(0, 1) draws
//!   (≡ `Beta(1, k)` in the standard parametrization).
//!
//! The resampling algorithm is from Massey (ICASSP 2008); the
//! sorted-uniforms recurrence is classical (Bentley & Saxe 1980;
//! Devroye 1986). See `README.md` for full proofs.
//!
//! # `f32`-only API
//!
//! All public APIs are `f32`. The realistic deployment target is the
//! Cortex-M4F (and similar single-precision FPUs), so the library
//! commits to single precision throughout. Where the obvious f32
//! algorithm would lose too much precision (the prefix-sum walks in
//! [`resample_indices`] and [`resample_indices_buffered`]), the
//! library uses Kahan compensated summation to recover O(ε) error
//! while staying entirely on the f32 FPU.
//!
//! # `no_std` support
//!
//! The library compiles in `no_std` mode. The crate has two mutually
//! exclusive math-source features:
//!
//! - `std` (default): use the standard library's libm bindings via
//!   the inherent `f32::powf` method.
//! - `libm`: use the [`libm`] crate via [`num_traits`] for `powf`.
//!   Suitable for bare-metal targets.
//!
//! Enable exactly one. The library performs no allocation: it
//! operates over caller-supplied slices (`&[f32]` for weights,
//! `&mut [u32]` for resample output) and never calls into
//! `alloc`.
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
