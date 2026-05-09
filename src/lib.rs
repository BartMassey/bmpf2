//! Sequential Importance Resampling (SIR) primitives, plus (forthcoming)
//! Bayesian Particle Filter (BPF) API on top.
//!
//! Today the crate exposes:
//!
//! - [`first_uniform`] (and its named-backend variants
//!   [`first_uniform_pow`] and [`first_uniform_rejection`]): a sample
//!   of the *minimum* of `k` iid Uniform(0, 1) draws — equivalently
//!   `Beta(1, k)` in the standard parametrization. This is the
//!   per-step variate that drives the linear-time order-statistic
//!   recurrence below.
//! - [`SortedUniforms`]: an iterator yielding `n` uniform variates in
//!   ascending order in O(n) time.
//! - [`resample_indices`]: linear-time perfect weighted resampling
//!   ("Method C" — streaming, no scratch buffer; one
//!   [`first_uniform`] call per output index).
//! - [`resample_indices_buffered`]: same statistical contract,
//!   buffered variant ("Method B" — uses Exp(1) variates via
//!   Gamma-ratio sorted uniforms; faster per-element on hardware
//!   with a slow `powf`, at the cost of a caller-supplied scratch
//!   buffer).
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
//!   the inherent `f32::ln` / `f32::powf` methods.
//! - `libm`: use the [`libm`] crate via [`num_traits`] for `ln` and
//!   `powf`. Suitable for bare-metal targets.
//!
//! Enable exactly one. The library performs no allocation: it
//! operates over caller-supplied slices (`&[f32]` for weights,
//! `&mut [usize]` for resample output, `&mut [f32]` for scratch) and
//! never calls into `alloc`.
//!
//! [`libm`]: https://crates.io/crates/libm
//! [`num_traits`]: https://crates.io/crates/num-traits
//!
//! # Backend selection
//!
//! The two backends for [`first_uniform`] are exposed unconditionally
//! as [`first_uniform_pow`] and [`first_uniform_rejection`] regardless
//! of feature flags, so callers can compare them or force a specific
//! one. The dispatcher [`first_uniform`] is feature-gated: with the
//! default `pow` feature it calls the pow backend; with only
//! `rejection` it calls rejection. With both, `pow` wins.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!(
    "Enable exactly one of the `std` or `libm` features so transcendental \
     math (`ln`, `powf`) is available."
);

#[cfg(not(any(feature = "pow", feature = "rejection")))]
compile_error!("At least one of the `pow` or `rejection` features must be enabled.");

mod first_uniform;
mod resample;
mod sorted_uniforms;

pub use first_uniform::{first_uniform, first_uniform_pow, first_uniform_rejection};
pub use resample::{resample_indices, resample_indices_buffered};
pub use sorted_uniforms::SortedUniforms;
