//! Microbenchmarks for `ltsis`. Run with `cargo bench`.
//!
//! Uses [Divan](https://docs.rs/divan): per-iteration timing with
//! outlier rejection and a tabular summary, no unstable-toolchain
//! requirement (we set `harness = false` in `Cargo.toml` so Divan's
//! `divan::main()` runs in place of the built-in `#[bench]` harness).
//!
//! # Bench groups
//!
//! - **`first_uniform`** — per-call cost of `first_uniform(rng, k)`.
//!   `divan::black_box`-fenced *between* calls so consecutive
//!   invocations cannot be fused by LLVM into a vectorized `powf`
//!   over a batch. The realistic streaming-sampler use is one-call-
//!   at-a-time inside the spacings recurrence (each call's output
//!   feeds the next call's `last`), so this scalar number is the
//!   relevant deployment cost.
//!
//! - **`pipeline_*_fenced`** — full sampler call (`sample_indices`
//!   collected into a buffer, or `sample_indices_buffered` writing
//!   to its slice argument), with `black_box` only at the *outer*
//!   API boundary. LLVM is free to autovectorize *inside* the
//!   sampler — and on x86 with AVX/SSE the buffered variant's two
//!   flat passes over `out` actually do vectorize. This is the
//!   number that reflects the compiler's optimization on the host.
//!
//! - **`pipeline_*_unfenced`** — same calls with no fences at all.
//!   LLVM may also fuse across iterations of the bench loop (e.g.
//!   constant-fold or hoist work). This is the upper-bound number
//!   on host x86; the gap to the fenced version measures how much
//!   internal autovectorization buys you on this CPU.
//!
//! Together fenced and unfenced bracket the realistic deployment
//! cost on the host. On scalar-only targets (Cortex-M4F) neither
//! number is directly relevant; bench on the real target.

use divan::{black_box, Bencher};
use ltsis::{first_uniform, sample_indices, sample_indices_buffered};
use rand::rngs::SmallRng;
use rand::SeedableRng;

fn main() {
    divan::main();
}

// ---------------------------------------------------------------------------
// first_uniform per-call
// ---------------------------------------------------------------------------

#[divan::bench(args = [2u32, 5, 10, 50, 200, 1000])]
fn first_uniform_per_call(bencher: Bencher, k: u32) {
    let mut rng = SmallRng::seed_from_u64(0xBEEF);
    bencher.bench_local(|| first_uniform(black_box(&mut rng), black_box(k)));
}

// ---------------------------------------------------------------------------
// Full sampling pipeline. Each variant is benched at five sizes
// spanning four orders of magnitude.
// ---------------------------------------------------------------------------

const PIPELINE_SIZES: &[usize] = &[100, 1_000, 10_000, 100_000, 1_000_000];

/// Streaming, fenced at the API boundary: LLVM sees `black_box` on
/// the inputs but is free to optimize inside `sample_indices`. The
/// streaming variant has a data-dependent recurrence so internal
/// autovectorization is largely unavailable; expect the unfenced
/// variant below to land in the same ballpark.
#[divan::bench(args = PIPELINE_SIZES)]
fn pipeline_streaming_fenced(bencher: Bencher, m: usize) {
    let weights: Vec<f32> = (1..=m).map(|x| x as f32).collect();
    let mut rng = SmallRng::seed_from_u64(0x1234);
    let mut out = vec![0u32; m];
    let n = m as u32;
    bencher.bench_local(|| {
        for (slot, j) in black_box(&mut out)
            .iter_mut()
            .zip(sample_indices(black_box(&mut rng), black_box(&weights), black_box(n)))
        {
            *slot = j;
        }
    });
}

/// Streaming, no fences: upper-bound number on this host.
#[divan::bench(args = PIPELINE_SIZES)]
fn pipeline_streaming_unfenced(bencher: Bencher, m: usize) {
    let weights: Vec<f32> = (1..=m).map(|x| x as f32).collect();
    let mut rng = SmallRng::seed_from_u64(0x1234);
    let mut out = vec![0u32; m];
    let n = m as u32;
    bencher.bench_local(|| {
        for (slot, j) in out.iter_mut().zip(sample_indices(&mut rng, &weights, n)) {
            *slot = j;
        }
    });
}

/// Buffered, fenced at the API boundary: internal phase-1 / phase-2
/// passes over `out` may autovectorize on x86 AVX/SSE.
#[divan::bench(args = PIPELINE_SIZES)]
fn pipeline_buffered_fenced(bencher: Bencher, m: usize) {
    let weights: Vec<f32> = (1..=m).map(|x| x as f32).collect();
    let mut rng = SmallRng::seed_from_u64(0x1234);
    let mut out = vec![0u32; m];
    bencher.bench_local(|| {
        sample_indices_buffered(black_box(&mut rng), black_box(&weights), black_box(&mut out));
    });
}

/// Buffered, no fences: LLVM is free to do anything across iterations
/// of the bench loop in addition to internal vectorization. The gap
/// from `pipeline_buffered_fenced` to here is small in practice (the
/// outer call is large enough that cross-call fusion isn't profitable)
/// but it bounds the question.
#[divan::bench(args = PIPELINE_SIZES)]
fn pipeline_buffered_unfenced(bencher: Bencher, m: usize) {
    let weights: Vec<f32> = (1..=m).map(|x| x as f32).collect();
    let mut rng = SmallRng::seed_from_u64(0x1234);
    let mut out = vec![0u32; m];
    bencher.bench_local(|| {
        sample_indices_buffered(&mut rng, &weights, &mut out);
    });
}
