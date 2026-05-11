//! Microbenchmarks for `ltsis`. Run with `cargo bench`.
//!
//! Uses `harness = false` (see `Cargo.toml`) so this is a regular
//! `fn main()` rather than the unstable `#[bench]` harness; we do
//! our own timing with `std::hint::black_box` fences at every
//! iteration boundary.
//!
//! Why fences: a naive accumulating loop on x86 with a SIMD libm
//! lets LLVM emit a vectorized `powf` over runs of consecutive
//! iterations — measuring batched throughput rather than scalar
//! per-call cost. Cortex-M4F (and any other no-SIMD target) can't
//! achieve that batched throughput, so the relevant number for
//! deciding-between-implementations is the *scalar* per-call cost.
//! `black_box` at every call boundary defeats the cross-iteration
//! vectorization without affecting the libm's own internal
//! micro-architecture (which is a property of the libm, not the
//! call site).

use ltsis::{first_uniform, sample_indices, sample_indices_buffered};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    println!("=== ltsis microbenchmark ===\n");
    bench_first_uniform();
    println!();
    bench_sample();
}

fn bench_first_uniform() {
    println!("[Bench] first_uniform per-call cost (black_box-fenced, 1M samples per k)");
    println!("  {:>5}  {:>16}", "k", "ns/sample");
    let n: u64 = 1_000_000;
    for &k in &[2u32, 5, 10, 50, 200, 1000] {
        let ns = bench_one(black_box(k), n, first_uniform);
        println!("  {k:5}  {ns:16.2}");
    }
    println!();
    println!("  Caveats:");
    println!("    - Numbers are host-specific. A well-tuned x86 libm computes");
    println!("      scalar `powf` in ~10 ns; Cortex-M scalar `powf` is ~30 ns");
    println!("      or more. Benchmark on the real target before drawing");
    println!("      conclusions.");
}

fn bench_sample() {
    println!("[Bench] Full sampling pipeline (m = n)");
    println!(
        "  {:>8}  {:>14}  {:>14}  {:>14}  {:>14}  {:>10}",
        "m = n", "C ns/call", "C ns/step", "B ns/call", "B ns/step", "C/B"
    );

    for &m in &[100usize, 1_000, 10_000, 100_000, 1_000_000] {
        let weights: Vec<f32> = (1..=m).map(|x| x as f32).collect();
        let n = m;
        let mut out = vec![0u32; n];

        let n_runs = ((30_000_000 / (m + n)).max(3)) as u64;

        // Streaming.
        let mut rng_c = SmallRng::seed_from_u64(0x1234);
        for _ in 0..3 {
            sample_indices(&mut rng_c, &weights, &mut out);
        }
        let t0 = Instant::now();
        for _ in 0..n_runs {
            sample_indices(
                black_box(&mut rng_c),
                black_box(&weights),
                black_box(&mut out),
            );
        }
        let elapsed_c = t0.elapsed();
        let ns_call_c = elapsed_c.as_nanos() as f64 / n_runs as f64;
        let ns_step_c = ns_call_c / (m + n) as f64;

        // Buffered.
        let mut rng_b = SmallRng::seed_from_u64(0x1234);
        for _ in 0..3 {
            sample_indices_buffered(&mut rng_b, &weights, &mut out);
        }
        let t0 = Instant::now();
        for _ in 0..n_runs {
            sample_indices_buffered(
                black_box(&mut rng_b),
                black_box(&weights),
                black_box(&mut out),
            );
        }
        let elapsed_b = t0.elapsed();
        let ns_call_b = elapsed_b.as_nanos() as f64 / n_runs as f64;
        let ns_step_b = ns_call_b / (m + n) as f64;

        println!(
            "  {:>8}  {:>14.0}  {:>14.2}  {:>14.0}  {:>14.2}  {:>9.2}x",
            m,
            ns_call_c,
            ns_step_c,
            ns_call_b,
            ns_step_b,
            ns_call_c / ns_call_b
        );
    }
}

/// Run a fenced microbenchmark. Returns ns/sample.
fn bench_one<Func>(k: u32, n: u64, mut f: Func) -> f64
where
    Func: FnMut(&mut SmallRng, u32) -> f32,
{
    let mut rng = SmallRng::seed_from_u64(0xBEEF);

    // Warmup.
    let mut s = 0.0_f32;
    for _ in 0..10_000 {
        s += f(&mut rng, k);
    }
    black_box(s);

    let t0 = Instant::now();
    let mut acc = 0.0_f32;
    for _ in 0..n {
        let kk = black_box(k);
        let x = f(black_box(&mut rng), kk);
        acc = black_box(acc + x);
    }
    let elapsed = t0.elapsed();
    black_box(acc);

    elapsed.as_nanos() as f64 / n as f64
}
