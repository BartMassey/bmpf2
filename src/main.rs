//! Test driver for the Beta(k, 1) sampler.
//!
//! Both `beta_k_1_pow` and `beta_k_1_rejection` are exercised regardless
//! of which feature is selected; the driver uses them directly rather
//! than going through the feature-gated `beta_k_1` dispatch.
//!
//! Each correctness test runs once for `f32` and once for `f64`. The
//! statistical machinery (KS distance, chi-squared) is computed in `f64`
//! regardless of which sample type is being tested, so the precision of
//! the test itself does not depend on the type under test.
//!
//! Microbench notes: `std::hint::black_box` is applied at every iteration
//! boundary to prevent LLVM from fusing or vectorizing across calls. This
//! is important because the goal is to compare per-call cost on hardware
//! that lacks SIMD (Cortex-M), not to measure peak vectorized throughput
//! on the host. Numbers will still be host-specific, but the *ratio*
//! between the two methods should be more representative.

use beta_k1::{
    beta_k_1_max_of_uniforms, beta_k_1_pow, beta_k_1_rejection, resample_indices,
    resample_indices_buffered, verify_acceptance_bound, BetaFloat, SortedUniforms,
};
use num_traits::Float;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::hint::black_box;
use std::time::Instant;

const KS_CRITICAL_001: f64 = 1.949; // 0.1% significance critical-value coefficient

fn main() {
    println!("=== Beta(k, 1) sampler test driver ===\n");

    let mut all_passed = true;

    // Test 1 is a numerical sanity check on the rejection-sampler constants;
    // running it for both float types adds nothing because the math is
    // float-type-independent. Run once in f64.
    all_passed &= test_acceptance_bound();
    println!();

    // Tests that exercise the actual implementation: run once per type.
    all_passed &= run_typed_tests::<f32>("f32");
    all_passed &= run_typed_tests::<f64>("f64");

    bench();

    println!(
        "\n=== Result: {} ===",
        if all_passed {
            "ALL TESTS PASSED"
        } else {
            "FAILURES"
        }
    );
    if !all_passed {
        std::process::exit(1);
    }
}

/// Cast a `BetaFloat` value to `f64` for use in statistical computations.
#[inline]
fn to_f64<F: Float>(x: F) -> f64 {
    x.to_f64().unwrap()
}

/// Run all per-type tests for a given float type.
///
/// Tests 8 and 9 are run twice — once each for the streaming
/// (`resample_indices`) and buffered (`resample_indices_buffered`)
/// resamplers — using a closure to abstract over the call shape.
fn run_typed_tests<F: BetaFloat>(label: &str) -> bool {
    let mut ok = true;
    ok &= test_range::<F>(label);
    println!();
    ok &= test_moments::<F>(label);
    println!();
    ok &= test_ks_against_theory::<F>(label);
    println!();
    ok &= test_ks_against_max_oracle::<F>(label);
    println!();
    ok &= test_sorted_uniforms_moments::<F>(label);
    println!();
    ok &= test_sorted_uniforms_pooled_ks::<F>(label);
    println!();
    ok &= test_resample_marginals::<F, _>(label, "streaming", |rng, w, o| {
        resample_indices::<F, _>(rng, w, o)
    });
    println!();
    ok &= test_resample_marginals::<F, _>(label, "buffered", |rng, w, o| {
        let mut scratch = vec![F::zero(); o.len()];
        resample_indices_buffered::<F, _>(rng, w, o, &mut scratch)
    });
    println!();
    ok &= test_resample_vs_multinomial::<F, _>(label, "streaming", |rng, w, o| {
        resample_indices::<F, _>(rng, w, o)
    });
    println!();
    ok &= test_resample_vs_multinomial::<F, _>(label, "buffered", |rng, w, o| {
        let mut scratch = vec![F::zero(); o.len()];
        resample_indices_buffered::<F, _>(rng, w, o, &mut scratch)
    });
    println!();
    ok
}

/// Test 1: A_k(Y) ≤ 1 across the support, i.e. M_k is correctly computed.
///
/// Numerical-only: the same formula is used regardless of float type, so
/// running this for both f32 and f64 wouldn't tell us anything new about
/// the algorithm. Run in f64 once.
fn test_acceptance_bound() -> bool {
    println!("[Test 1] log A_k(Y) ≤ 0 across the support (M_k is correct sup) [f64]");
    let mut all_ok = true;
    for k in 2..=200 {
        let max_log_accept: f64 = verify_acceptance_bound(k, 100_000);
        if max_log_accept > 1e-12 {
            println!("  k={:4}: max log A_k(Y) = {:+.3e} > 0", k, max_log_accept);
            all_ok = false;
        }
    }
    if all_ok {
        println!("  k = 2..=200: max log A_k(Y) ≤ 0 across all tested grids");
    }
    all_ok
}

/// Test 2: All samples are in [0, 1] for both implementations.
fn test_range<F: BetaFloat>(label: &str) -> bool {
    println!(
        "[Test 2] Samples are within [0, 1] (both implementations) [{}]",
        label
    );
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut all_ok = true;
    let zero = F::zero();
    let one = F::one();
    for &k in &[1u32, 2, 3, 5, 10, 100, 1000] {
        for _ in 0..100_000 {
            let x_pow: F = beta_k_1_pow(&mut rng, k);
            let x_rej: F = beta_k_1_rejection(&mut rng, k);
            if x_pow.is_nan() || x_pow < zero || x_pow > one {
                println!("  pow:       k={}: out-of-range {:+e}", k, to_f64(x_pow));
                all_ok = false;
                break;
            }
            if x_rej.is_nan() || x_rej < zero || x_rej > one {
                println!("  rejection: k={}: out-of-range {:+e}", k, to_f64(x_rej));
                all_ok = false;
                break;
            }
        }
    }
    if all_ok {
        println!("  k ∈ {{1,2,3,5,10,100,1000}}: 100k samples each, both implementations OK");
    }
    all_ok
}

/// Test 3: Empirical moments match closed-form Beta(k, 1) values.
///
/// For X ~ Beta(k, 1):  `E[X] = k/(k+1)`,  `Var[X] = k / [(k+1)^2 (k+2)]`.
fn test_moments<F: BetaFloat>(label: &str) -> bool {
    println!(
        "[Test 3] Empirical moments match Beta(k, 1) closed form (rejection impl) [{}]",
        label
    );
    let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
    let mut all_ok = true;
    let n_samples = 1_000_000;

    for &k in &[1u32, 2, 3, 5, 10, 50, 200] {
        let kf = k as f64;
        let theoretical_mean = kf / (kf + 1.0);
        let theoretical_var = kf / ((kf + 1.0).powi(2) * (kf + 2.0));

        // Accumulate in f64 to keep stat noise from float type under test
        // out of the test's own machinery.
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        for _ in 0..n_samples {
            let x: F = beta_k_1_rejection(&mut rng, k);
            let xf = to_f64(x);
            sum += xf;
            sum_sq += xf * xf;
        }
        let mean = sum / n_samples as f64;
        let var = sum_sq / n_samples as f64 - mean * mean;

        let se_mean = (theoretical_var / n_samples as f64).sqrt();
        let mean_err = (mean - theoretical_mean).abs();
        let mean_ok = mean_err < 5.0 * se_mean;
        let var_rel_err = (var - theoretical_var).abs() / theoretical_var;
        let var_ok = var_rel_err < 0.02;

        let status = if mean_ok && var_ok { "OK" } else { "FAIL" };
        println!(
            "  k={:4}: mean {:.6} (theo {:.6}, err {:+.2e}, 5σ={:+.2e})  var rel-err {:.2e}  [{}]",
            k,
            mean,
            theoretical_mean,
            mean - theoretical_mean,
            5.0 * se_mean,
            var_rel_err,
            status
        );
        if !mean_ok || !var_ok {
            all_ok = false;
        }
    }
    all_ok
}

/// Test 4: KS test of the rejection sampler against the analytic CDF F_k(x) = x^k.
fn test_ks_against_theory<F: BetaFloat>(label: &str) -> bool {
    println!(
        "[Test 4] KS test of rejection sampler vs. F_k(x) = x^k [{}]",
        label
    );
    let mut rng = StdRng::seed_from_u64(0x12345678);
    let mut all_ok = true;
    let n = 50_000;

    for &k in &[1u32, 2, 3, 5, 10, 50, 200] {
        // Cast samples to f64 for sorting and KS-statistic computation.
        let mut samples: Vec<f64> = (0..n)
            .map(|_| to_f64(beta_k_1_rejection::<F, _>(&mut rng, k)))
            .collect();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let kf = k as f64;
        let mut d_max: f64 = 0.0;
        for (i, &x) in samples.iter().enumerate() {
            let f_emp_above = (i + 1) as f64 / n as f64;
            let f_emp_below = i as f64 / n as f64;
            let f_theory = x.powf(kf);
            let d = (f_emp_above - f_theory)
                .abs()
                .max((f_theory - f_emp_below).abs());
            if d > d_max {
                d_max = d;
            }
        }

        let critical = KS_CRITICAL_001 / (n as f64).sqrt();
        let pass = d_max < critical;
        let status = if pass { "OK" } else { "FAIL" };
        println!(
            "  k={:4}: D = {:.5}, critical (0.1%) = {:.5}  [{}]",
            k, d_max, critical, status
        );
        if !pass {
            all_ok = false;
        }
    }
    all_ok
}

/// Test 5: Two-sample KS — rejection sampler vs. the max-of-k-uniforms oracle.
fn test_ks_against_max_oracle<F: BetaFloat>(label: &str) -> bool {
    println!(
        "[Test 5] Two-sample KS: rejection vs. max-of-k-uniforms oracle [{}]",
        label
    );
    let mut rng_a = StdRng::seed_from_u64(0xAAAA_AAAA);
    let mut rng_b = StdRng::seed_from_u64(0xBBBB_BBBB);
    let mut all_ok = true;
    let n = 20_000;

    for &k in &[2u32, 3, 5, 10, 50] {
        let mut a: Vec<f64> = (0..n)
            .map(|_| to_f64(beta_k_1_rejection::<F, _>(&mut rng_a, k)))
            .collect();
        let mut b: Vec<f64> = (0..n)
            .map(|_| to_f64(beta_k_1_max_of_uniforms::<F, _>(&mut rng_b, k)))
            .collect();
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        b.sort_by(|x, y| x.partial_cmp(y).unwrap());

        let d_max = two_sample_ks(&a, &b);
        let critical = KS_CRITICAL_001 * ((2 * n) as f64 / (n * n) as f64).sqrt();
        let pass = d_max < critical;
        let status = if pass { "OK" } else { "FAIL" };
        println!(
            "  k={:3}: D = {:.5}, critical (0.1%) = {:.5}  [{}]",
            k, d_max, critical, status
        );
        if !pass {
            all_ok = false;
        }
    }
    all_ok
}

fn two_sample_ks(a: &[f64], b: &[f64]) -> f64 {
    let mut i = 0;
    let mut j = 0;
    let mut d_max: f64 = 0.0;
    let na = a.len() as f64;
    let nb = b.len() as f64;
    while i < a.len() && j < b.len() {
        if a[i] <= b[j] {
            i += 1;
        } else {
            j += 1;
        }
        let d = (i as f64 / na - j as f64 / nb).abs();
        if d > d_max {
            d_max = d;
        }
    }
    d_max
}

// ===========================================================================
// Resampling tests
// ===========================================================================

/// Test 6: Empirical mean and variance of each order statistic position
/// match closed-form values.
///
/// For the i-th of n sorted uniforms (1-indexed):
///   E[U_(i)]   = i / (n + 1)
///   Var[U_(i)] = i (n - i + 1) / [(n + 1)^2 (n + 2)]
fn test_sorted_uniforms_moments<F: BetaFloat>(label: &str) -> bool {
    println!(
        "[Test 6] Sorted uniforms: per-position moments match closed form [{}]",
        label
    );
    let mut rng = StdRng::seed_from_u64(0xABCD_1234);
    let mut all_ok = true;

    for &n in &[5u32, 20, 100] {
        let n_runs = 200_000;

        let mut sum = vec![0.0_f64; n as usize];
        let mut sum_sq = vec![0.0_f64; n as usize];

        for _ in 0..n_runs {
            let mut iter = SortedUniforms::<F, _>::new(&mut rng, n);
            for i in 0..n as usize {
                let v = to_f64(iter.next().unwrap());
                sum[i] += v;
                sum_sq[i] += v * v;
            }
        }

        let mut max_mean_z: f64 = 0.0;
        let mut max_var_rel: f64 = 0.0;
        let mut worst_pos: usize = 0;
        for i in 0..n as usize {
            let i1 = (i + 1) as f64;
            let nf = n as f64;
            let theo_mean = i1 / (nf + 1.0);
            let theo_var = i1 * (nf - i1 + 1.0) / ((nf + 1.0).powi(2) * (nf + 2.0));

            let emp_mean = sum[i] / n_runs as f64;
            let emp_var = sum_sq[i] / n_runs as f64 - emp_mean * emp_mean;

            let se_mean = (theo_var / n_runs as f64).sqrt();
            let z = (emp_mean - theo_mean).abs() / se_mean;
            let rel = (emp_var - theo_var).abs() / theo_var;

            if z > max_mean_z {
                max_mean_z = z;
                worst_pos = i + 1;
            }
            if rel > max_var_rel {
                max_var_rel = rel;
            }
        }

        let mean_ok = max_mean_z < 5.0;
        let var_ok = max_var_rel < 0.05;
        let status = if mean_ok && var_ok { "OK" } else { "FAIL" };
        println!(
            "  n={:4}: worst position {:>3}, mean z = {:.2}σ, var rel-err ≤ {:.2e}  [{}]",
            n, worst_pos, max_mean_z, max_var_rel, status
        );
        if !mean_ok || !var_ok {
            all_ok = false;
        }
    }
    all_ok
}

/// Test 7: Pooled KS test of sorted-uniforms output against U(0, 1).
fn test_sorted_uniforms_pooled_ks<F: BetaFloat>(label: &str) -> bool {
    println!(
        "[Test 7] Sorted uniforms: pooled output is Uniform(0, 1) [{}]",
        label
    );
    let mut rng = StdRng::seed_from_u64(0xCAFE_F00D);
    let mut all_ok = true;

    for &n in &[10u32, 100, 1000] {
        let n_runs = 50_000 / n.max(1) as usize;
        let total = n_runs * n as usize;
        let mut pooled: Vec<f64> = Vec::with_capacity(total);
        for _ in 0..n_runs {
            let iter = SortedUniforms::<F, _>::new(&mut rng, n);
            pooled.extend(iter.map(to_f64));
        }
        pooled.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut d_max: f64 = 0.0;
        let len = pooled.len() as f64;
        for (i, &x) in pooled.iter().enumerate() {
            let f_above = (i + 1) as f64 / len;
            let f_below = i as f64 / len;
            let d = (f_above - x).abs().max((x - f_below).abs());
            if d > d_max {
                d_max = d;
            }
        }
        let critical = KS_CRITICAL_001 / len.sqrt();
        let pass = d_max < critical;
        let status = if pass { "OK" } else { "FAIL" };
        println!(
            "  n={:4}, pooled={:6}: D = {:.5}, critical (0.1%) = {:.5}  [{}]",
            n, total, d_max, critical, status
        );
        if !pass {
            all_ok = false;
        }
    }
    all_ok
}

/// Construct an `F`-typed weight vector from f64 literals.
fn weights_from_f64<F: BetaFloat>(ws: &[f64]) -> Vec<F> {
    ws.iter().map(|&w| F::from(w).unwrap()).collect()
}

/// Test 8: Marginal index probabilities under the supplied resampler
/// match the weight-proportional probabilities, by chi-squared
/// goodness-of-fit. Runs once per resampler (streaming, buffered).
fn test_resample_marginals<F: BetaFloat, Resampler>(
    label: &str,
    method: &str,
    mut resample: Resampler,
) -> bool
where
    Resampler: FnMut(&mut StdRng, &[F], &mut [usize]),
{
    println!(
        "[Test 8] Resampling: index marginal probabilities (chi-squared) [{}/{}]",
        label, method
    );
    let mut rng = StdRng::seed_from_u64(0xFEED_BEEF);
    let mut all_ok = true;

    let test_cases: Vec<(&str, Vec<f64>)> = vec![
        ("uniform-10", vec![1.0; 10]),
        ("decreasing-8", (1..=8).rev().map(|x| x as f64).collect()),
        (
            "peaky-12",
            (0..12)
                .map(|i| if i == 5 { 50.0 } else { 1.0 })
                .collect::<Vec<_>>(),
        ),
        ("zeroes-mixed-6", vec![0.0, 2.0, 0.0, 1.0, 3.0, 4.0]),
    ];

    for (name, weights_f64) in &test_cases {
        let weights: Vec<F> = weights_from_f64(weights_f64);
        let m = weights.len();
        let total: f64 = weights_f64.iter().sum();
        let n_per_run = 50;
        let n_runs = 4_000;
        let total_draws = n_per_run * n_runs;

        let mut counts = vec![0u64; m];
        let mut buf = vec![0usize; n_per_run];
        for _ in 0..n_runs {
            resample(&mut rng, &weights, &mut buf);
            for &idx in &buf {
                counts[idx] += 1;
            }
        }

        let mut chi_sq = 0.0;
        let mut dof = 0_i32;
        let mut zero_violation = false;
        for i in 0..m {
            let expected = total_draws as f64 * weights_f64[i] / total;
            if expected == 0.0 {
                if counts[i] != 0 {
                    zero_violation = true;
                }
                continue;
            }
            let diff = counts[i] as f64 - expected;
            chi_sq += diff * diff / expected;
            dof += 1;
        }
        dof -= 1;

        let z = 3.090;
        let dofs = dof as f64;
        let crit = dofs * (1.0 - 2.0 / (9.0 * dofs) + z * (2.0 / (9.0 * dofs)).sqrt()).powi(3);
        let pass = !zero_violation && chi_sq < crit;
        let status = if pass { "OK" } else { "FAIL" };
        println!(
            "  {:>16}: χ² = {:7.2}, dof = {}, critical (0.1%) ≈ {:7.2}  [{}]",
            name, chi_sq, dof, crit, status
        );
        if !pass {
            all_ok = false;
        }
    }
    all_ok
}

/// Test 9: Resampling matches naive multinomial sampling, by two-sample
/// chi-squared on the index-count vectors. Runs once per resampler.
fn test_resample_vs_multinomial<F: BetaFloat, Resampler>(
    label: &str,
    method: &str,
    mut resample: Resampler,
) -> bool
where
    Resampler: FnMut(&mut StdRng, &[F], &mut [usize]),
{
    println!(
        "[Test 9] Resampling matches naive multinomial (two-sample χ²) [{}/{}]",
        label, method
    );
    let mut rng_a = StdRng::seed_from_u64(0xA1A1_A1A1);
    let mut rng_b = StdRng::seed_from_u64(0xB2B2_B2B2);
    let mut all_ok = true;

    let test_cases: Vec<(&str, Vec<f64>)> = vec![
        ("uniform-10", vec![1.0; 10]),
        ("skewed-8", (1..=8).map(|x| x as f64).collect()),
        (
            "peaky-15",
            (0..15).map(|i| if i == 7 { 30.0 } else { 1.0 }).collect(),
        ),
    ];

    for (name, weights_f64) in &test_cases {
        let weights: Vec<F> = weights_from_f64(weights_f64);
        let m = weights.len();
        let n_per_run = 100;
        let n_runs = 2_000;

        let mut counts_a = vec![0u64; m];
        let mut counts_b = vec![0u64; m];
        let mut buf = vec![0usize; n_per_run];

        for _ in 0..n_runs {
            resample(&mut rng_a, &weights, &mut buf);
            for &idx in &buf {
                counts_a[idx] += 1;
            }
            naive_multinomial::<F, _>(&mut rng_b, &weights, &mut buf);
            for &idx in &buf {
                counts_b[idx] += 1;
            }
        }

        let n_a = counts_a.iter().sum::<u64>() as f64;
        let n_b = counts_b.iter().sum::<u64>() as f64;
        let n_total = n_a + n_b;

        let mut chi_sq = 0.0;
        let mut dof = 0_i32;
        for i in 0..m {
            let row_total = counts_a[i] + counts_b[i];
            if row_total == 0 {
                continue;
            }
            let e_a = n_a * row_total as f64 / n_total;
            let e_b = n_b * row_total as f64 / n_total;
            let d_a = counts_a[i] as f64 - e_a;
            let d_b = counts_b[i] as f64 - e_b;
            chi_sq += d_a * d_a / e_a + d_b * d_b / e_b;
            dof += 1;
        }
        dof -= 1;

        let z = 3.090;
        let dofs = dof as f64;
        let crit = dofs * (1.0 - 2.0 / (9.0 * dofs) + z * (2.0 / (9.0 * dofs)).sqrt()).powi(3);
        let pass = chi_sq < crit;
        let status = if pass { "OK" } else { "FAIL" };
        println!(
            "  {:>16}: χ² = {:7.2}, dof = {}, critical (0.1%) ≈ {:7.2}  [{}]",
            name, chi_sq, dof, crit, status
        );
        if !pass {
            all_ok = false;
        }
    }
    all_ok
}

/// Naive multinomial resampler: for each output, draw a uniform on
/// [0, total) and binary-search the cumulative-weight array.
/// O(m + n log m). Used as the trusted reference in Test 9.
fn naive_multinomial<F: BetaFloat, R: Rng + ?Sized>(rng: &mut R, weights: &[F], out: &mut [usize]) {
    let mut cum = vec![F::zero(); weights.len()];
    let mut t = F::zero();
    for (i, &w) in weights.iter().enumerate() {
        t = t + w;
        cum[i] = t;
    }
    for slot in out.iter_mut() {
        let target: F = F::sample_uniform(rng) * t;
        // partition_point: first index with cum[idx] > target
        let idx = cum.partition_point(|&c| c <= target);
        *slot = idx.min(weights.len() - 1);
    }
}

/// Microbenchmark with `black_box` fences at every call boundary to defeat
/// LLVM autovectorization, fusion, and inlining-driven cross-call CSE.
fn bench() {
    println!("[Bench] Per-call cost (black_box-fenced, 1M samples per k)");
    bench_typed::<f64>("f64");
    println!();
    bench_typed::<f32>("f32");
    println!();
    println!("  Caveats:");
    println!("    - Numbers are host-specific. On a host with a fast SIMD libm,");
    println!("      `pow` benefits more than `rejection` does, even with black_box.");
    println!("    - Cortex-M targets have no SIMD and a slower scalar `powf`;");
    println!("      benchmark on the real target before drawing conclusions.");

    println!();
    bench_resample_typed::<f64>("f64");
    println!();
    bench_resample_typed::<f32>("f32");
}

fn bench_typed<F: BetaFloat>(label: &str) {
    let n: u64 = 1_000_000;

    println!(
        "  [{}]  {:>5}  {:>16}  {:>16}  {:>10}",
        label, "k", "pow (ns/sample)", "rejection (ns/sample)", "rej/pow"
    );

    for &k in &[2u32, 5, 10, 50, 200, 1000] {
        let pow_ns = bench_one(black_box(k), n, |rng, kk| beta_k_1_pow::<F, _>(rng, kk));
        let rej_ns = bench_one(black_box(k), n, |rng, kk| {
            beta_k_1_rejection::<F, _>(rng, kk)
        });
        println!(
            "  [{}]  {:5}  {:16.2}  {:16.2}  {:9.2}x",
            label,
            k,
            pow_ns,
            rej_ns,
            rej_ns / pow_ns
        );
    }
}

fn bench_resample_typed<F: BetaFloat>(label: &str) {
    println!("[Bench] Full resampling pipeline (m = n) [{}]", label);
    println!(
        "  {:>8}  {:>14}  {:>14}  {:>14}  {:>14}  {:>10}",
        "m = n", "C ns/call", "C ns/step", "B ns/call", "B ns/step", "C/B"
    );

    for &m in &[100usize, 1_000, 10_000, 100_000] {
        let weights: Vec<F> = (1..=m).map(|x| F::from(x).unwrap()).collect();
        let n = m;
        let mut out = vec![0usize; n];
        let mut scratch = vec![F::zero(); n];

        let n_runs = ((30_000_000 / (m + n)).max(3)) as u64;

        // Streaming (Method C).
        let mut rng_c = StdRng::seed_from_u64(0x1234);
        for _ in 0..3 {
            resample_indices::<F, _>(&mut rng_c, &weights, &mut out);
        }
        let t0 = Instant::now();
        for _ in 0..n_runs {
            resample_indices::<F, _>(
                black_box(&mut rng_c),
                black_box(&weights),
                black_box(&mut out),
            );
        }
        let elapsed_c = t0.elapsed();
        let ns_call_c = elapsed_c.as_nanos() as f64 / n_runs as f64;
        let ns_step_c = ns_call_c / (m + n) as f64;

        // Buffered (Method B).
        let mut rng_b = StdRng::seed_from_u64(0x1234);
        for _ in 0..3 {
            resample_indices_buffered::<F, _>(&mut rng_b, &weights, &mut out, &mut scratch);
        }
        let t0 = Instant::now();
        for _ in 0..n_runs {
            resample_indices_buffered::<F, _>(
                black_box(&mut rng_b),
                black_box(&weights),
                black_box(&mut out),
                black_box(&mut scratch),
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
fn bench_one<F, Func>(k: u32, n: u64, mut f: Func) -> f64
where
    F: BetaFloat,
    Func: FnMut(&mut StdRng, u32) -> F,
{
    let mut rng = StdRng::seed_from_u64(0xBEEF);

    let mut s = F::zero();
    for _ in 0..10_000 {
        s = s + f(&mut rng, k);
    }
    black_box(s);

    let t0 = Instant::now();
    let mut acc = F::zero();
    for _ in 0..n {
        let kk = black_box(k);
        let x = f(black_box(&mut rng), kk);
        acc = black_box(acc + x);
    }
    let elapsed = t0.elapsed();
    black_box(acc);

    elapsed.as_nanos() as f64 / n as f64
}
