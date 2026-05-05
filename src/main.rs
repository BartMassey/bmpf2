//! Test driver for the Beta(k, 1) sampler.
//!
//! Both `beta_k_1_pow` and `beta_k_1_rejection` are exercised regardless
//! of which feature is selected; the driver uses them directly rather
//! than going through the feature-gated `beta_k_1` dispatch.
//!
//! Microbench notes: `std::hint::black_box` is applied at every iteration
//! boundary to prevent LLVM from fusing or vectorizing across calls. This
//! is important because the goal is to compare per-call cost on hardware
//! that lacks SIMD (Cortex-M), not to measure peak vectorized throughput
//! on the host. Numbers will still be host-specific, but the *ratio*
//! between the two methods should be more representative.

use beta_k1::{
    beta_k_1_max_of_uniforms, beta_k_1_pow, beta_k_1_rejection, resample_indices, verify_acceptance_bound,
    SortedUniforms,
};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::hint::black_box;
use std::time::Instant;

const KS_CRITICAL_001: f64 = 1.949; // 0.1% significance critical-value coefficient

fn main() {
    println!("=== Beta(k, 1) sampler test driver ===\n");

    let mut all_passed = true;
    all_passed &= test_acceptance_bound();
    println!();
    all_passed &= test_range();
    println!();
    all_passed &= test_moments();
    println!();
    all_passed &= test_ks_against_theory();
    println!();
    all_passed &= test_ks_against_max_oracle();
    println!();
    all_passed &= test_sorted_uniforms_moments();
    println!();
    all_passed &= test_sorted_uniforms_pooled_ks();
    println!();
    all_passed &= test_resample_marginals();
    println!();
    all_passed &= test_resample_vs_multinomial();
    println!();
    bench();

    println!(
        "\n=== Result: {} ===",
        if all_passed { "ALL TESTS PASSED" } else { "FAILURES" }
    );
    if !all_passed {
        std::process::exit(1);
    }
}

/// Test 1: A_k(Y) ≤ 1 across the support, i.e. M_k is correctly computed.
fn test_acceptance_bound() -> bool {
    println!("[Test 1] log A_k(Y) ≤ 0 across the support (M_k is correct sup)");
    let mut all_ok = true;
    for k in 2..=200 {
        let max_log_accept = verify_acceptance_bound(k, 100_000);
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
fn test_range() -> bool {
    println!("[Test 2] Samples are within [0, 1] (both implementations)");
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut all_ok = true;
    for &k in &[1u32, 2, 3, 5, 10, 100, 1000] {
        for _ in 0..100_000 {
            let x_pow = beta_k_1_pow(&mut rng, k);
            let x_rej = beta_k_1_rejection(&mut rng, k);
            if !(0.0..=1.0).contains(&x_pow) || x_pow.is_nan() {
                println!("  pow:       k={}: out-of-range {}", k, x_pow);
                all_ok = false;
                break;
            }
            if !(0.0..=1.0).contains(&x_rej) || x_rej.is_nan() {
                println!("  rejection: k={}: out-of-range {}", k, x_rej);
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
/// For X ~ Beta(k, 1):  E[X] = k/(k+1),  Var[X] = k / [(k+1)^2 (k+2)].
fn test_moments() -> bool {
    println!("[Test 3] Empirical moments match Beta(k, 1) closed form (rejection impl)");
    let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
    let mut all_ok = true;
    let n_samples = 1_000_000;

    for &k in &[1u32, 2, 3, 5, 10, 50, 200] {
        let kf = k as f64;
        let theoretical_mean = kf / (kf + 1.0);
        let theoretical_var = kf / ((kf + 1.0).powi(2) * (kf + 2.0));

        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n_samples {
            let x = beta_k_1_rejection(&mut rng, k);
            sum += x;
            sum_sq += x * x;
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
fn test_ks_against_theory() -> bool {
    println!("[Test 4] KS test of rejection sampler vs. F_k(x) = x^k");
    let mut rng = StdRng::seed_from_u64(0x12345678);
    let mut all_ok = true;
    let n = 50_000;

    for &k in &[1u32, 2, 3, 5, 10, 50, 200] {
        let mut samples: Vec<f64> = (0..n).map(|_| beta_k_1_rejection(&mut rng, k)).collect();
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
fn test_ks_against_max_oracle() -> bool {
    println!("[Test 5] Two-sample KS: rejection vs. max-of-k-uniforms oracle");
    let mut rng_a = StdRng::seed_from_u64(0xAAAA_AAAA);
    let mut rng_b = StdRng::seed_from_u64(0xBBBB_BBBB);
    let mut all_ok = true;
    let n = 20_000;

    for &k in &[2u32, 3, 5, 10, 50] {
        let mut a: Vec<f64> = (0..n).map(|_| beta_k_1_rejection(&mut rng_a, k)).collect();
        let mut b: Vec<f64> = (0..n).map(|_| beta_k_1_max_of_uniforms(&mut rng_b, k)).collect();
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
fn test_sorted_uniforms_moments() -> bool {
    println!("[Test 6] Sorted uniforms: per-position moments match closed form");
    let mut rng = StdRng::seed_from_u64(0xABCD_1234);
    let mut all_ok = true;

    for &n in &[5u32, 20, 100] {
        let n_runs = 200_000;

        // Accumulate per-position sums and sums-of-squares across runs.
        let mut sum = vec![0.0_f64; n as usize];
        let mut sum_sq = vec![0.0_f64; n as usize];

        for _ in 0..n_runs {
            let mut iter = SortedUniforms::new(&mut rng, n);
            for i in 0..n as usize {
                let v = iter.next().unwrap();
                sum[i] += v;
                sum_sq[i] += v * v;
            }
        }

        // Check the worst position rather than printing all of them.
        let mut max_mean_z: f64 = 0.0;
        let mut max_var_rel: f64 = 0.0;
        let mut worst_pos: usize = 0;
        for i in 0..n as usize {
            let i1 = (i + 1) as f64; // 1-indexed
            let nf = n as f64;
            let theo_mean = i1 / (nf + 1.0);
            let theo_var =
                i1 * (nf - i1 + 1.0) / ((nf + 1.0).powi(2) * (nf + 2.0));

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
///
/// If the generator produces correct order statistics, then pooling all
/// values across many runs yields a sample of i.i.d. Uniform(0, 1) draws.
/// We KS-test the pooled empirical CDF against the uniform CDF.
fn test_sorted_uniforms_pooled_ks() -> bool {
    println!("[Test 7] Sorted uniforms: pooled output is Uniform(0, 1)");
    let mut rng = StdRng::seed_from_u64(0xCAFE_F00D);
    let mut all_ok = true;

    for &n in &[10u32, 100, 1000] {
        let n_runs = 50_000 / n.max(1) as usize; // Keep total samples ~ 50k
        let total = n_runs * n as usize;
        let mut pooled = Vec::with_capacity(total);
        for _ in 0..n_runs {
            let iter = SortedUniforms::new(&mut rng, n);
            pooled.extend(iter);
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

/// Test 8: Marginal index probabilities under `resample_indices` match
/// the weight-proportional probabilities, by chi-squared goodness-of-fit.
///
/// Run resampling many times against a fixed weight vector; tally how
/// often each index appears across all draws; chi-squared test against
/// expected counts.
fn test_resample_marginals() -> bool {
    println!("[Test 8] Resampling: index marginal probabilities (chi-squared)");
    let mut rng = StdRng::seed_from_u64(0xFEED_BEEF);
    let mut all_ok = true;

    // Several weight distributions: uniform, decreasing, peaky.
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

    for (name, weights) in &test_cases {
        let m = weights.len();
        let total: f64 = weights.iter().sum();
        let n_per_run = 50;
        let n_runs = 4_000;
        let total_draws = n_per_run * n_runs;

        // Tally counts.
        let mut counts = vec![0u64; m];
        let mut buf = vec![0usize; n_per_run];
        for _ in 0..n_runs {
            resample_indices(&mut rng, weights, &mut buf);
            for &idx in &buf {
                counts[idx] += 1;
            }
        }

        // Chi-squared. Lump zero-expected cells out (they should have count 0).
        let mut chi_sq = 0.0;
        let mut dof = 0_i32;
        let mut zero_violation = false;
        for i in 0..m {
            let expected = total_draws as f64 * weights[i] / total;
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
        dof -= 1; // we summed to a fixed total

        // Critical value at 0.1% significance for chi-squared. Approximate
        // with Wilson–Hilferty: chi^2 ~ dof * (1 - 2/(9 dof) + z * sqrt(2/(9 dof)))^3
        // with z = 3.090 for 0.1% upper tail.
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
/// chi-squared on the index-count vectors.
///
/// Run both methods many times against the same weight vector; pool counts;
/// chi-squared against the consistency of the two empirical distributions.
fn test_resample_vs_multinomial() -> bool {
    println!("[Test 9] Resampling matches naive multinomial (two-sample χ²)");
    let mut rng_a = StdRng::seed_from_u64(0xA1A1_A1A1);
    let mut rng_b = StdRng::seed_from_u64(0xB2B2_B2B2);
    let mut all_ok = true;

    let test_cases: Vec<(&str, Vec<f64>)> = vec![
        ("uniform-10", vec![1.0; 10]),
        ("skewed-8", (1..=8).map(|x| x as f64).collect()),
        ("peaky-15", (0..15).map(|i| if i == 7 { 30.0 } else { 1.0 }).collect()),
    ];

    for (name, weights) in &test_cases {
        let m = weights.len();
        let n_per_run = 100;
        let n_runs = 2_000;

        let mut counts_a = vec![0u64; m];
        let mut counts_b = vec![0u64; m];
        let mut buf = vec![0usize; n_per_run];

        for _ in 0..n_runs {
            resample_indices(&mut rng_a, weights, &mut buf);
            for &idx in &buf {
                counts_a[idx] += 1;
            }
            naive_multinomial(&mut rng_b, weights, &mut buf);
            for &idx in &buf {
                counts_b[idx] += 1;
            }
        }

        // Two-sample chi-squared (homogeneity test):
        //   χ² = Σ_i (O_a_i - E_a_i)^2 / E_a_i + (O_b_i - E_b_i)^2 / E_b_i
        // where E_a_i = N_a · (O_a_i + O_b_i) / N_total, similarly E_b_i,
        // with N_a = N_b = total draws per method.
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

        // Same Wilson–Hilferty critical-value approximation.
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
fn naive_multinomial<R: Rng + ?Sized>(rng: &mut R, weights: &[f64], out: &mut [usize]) {
    let mut cum = vec![0.0_f64; weights.len()];
    let mut t = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        t += w;
        cum[i] = t;
    }
    for slot in out.iter_mut() {
        let target: f64 = rng.gen::<f64>() * t;
        // binary search for first index with cum[idx] > target
        let idx = cum.partition_point(|&c| c <= target);
        *slot = idx.min(weights.len() - 1);
    }
}

/// Microbenchmark with `black_box` fences at every call boundary to defeat
/// LLVM autovectorization, fusion, and inlining-driven cross-call CSE.
///
/// On hardware with a vectorizable `pow` (modern x86), a naive benchmark
/// will have LLVM emit AVX `pow` over runs of consecutive iterations,
/// inflating the apparent throughput far beyond what a Cortex-M can do.
/// The fences below force one scalar call per iteration. The absolute
/// numbers are still host-specific; the *ratio* is more meaningful.
fn bench() {
    println!("[Bench] Per-call cost (black_box-fenced, 1M samples per k)");
    let n: u64 = 1_000_000;

    println!(
        "  {:>5}  {:>16}  {:>16}  {:>10}",
        "k", "pow (ns/sample)", "rejection (ns/sample)", "rej/pow"
    );

    for &k in &[2u32, 5, 10, 50, 200, 1000] {
        let pow_ns = bench_one(black_box(k), n, |rng, kk| beta_k_1_pow(rng, kk));
        let rej_ns = bench_one(black_box(k), n, |rng, kk| beta_k_1_rejection(rng, kk));
        println!(
            "  {:5}  {:16.2}  {:16.2}  {:9.2}x",
            k,
            pow_ns,
            rej_ns,
            rej_ns / pow_ns
        );
    }

    println!();
    println!("  Caveats:");
    println!("    - Numbers are host-specific. On a host with a fast SIMD libm,");
    println!("      `pow` benefits more than `rejection` does, even with black_box.");
    println!("    - Cortex-M targets have no SIMD and a slower scalar `powf`;");
    println!("      benchmark on the real target before drawing conclusions.");

    println!();
    bench_resample();
}

/// Resampling pipeline benchmark. Reports time per resample call and
/// time per (m + n)-step, where m = weights.len() and n = out.len().
/// Useful as a sanity check that the whole-pipeline cost scales linearly
/// and is dominated by the variate generation rather than something
/// quadratic.
fn bench_resample() {
    println!("[Bench] Full resampling pipeline (m = n)");
    println!(
        "  {:>8}  {:>14}  {:>14}",
        "m = n", "ns/call", "ns/step (m+n)"
    );

    for &m in &[100usize, 1_000, 10_000, 100_000] {
        let weights: Vec<f64> = (1..=m).map(|x| x as f64).collect();
        let n = m;
        let mut out = vec![0usize; n];
        let mut rng = StdRng::seed_from_u64(0x1234);

        // Warmup.
        for _ in 0..3 {
            resample_indices(&mut rng, &weights, &mut out);
        }

        // Choose run count to keep total work ~ 30M steps.
        let n_runs = ((30_000_000 / (m + n)).max(3)) as u64;
        let t0 = Instant::now();
        for _ in 0..n_runs {
            resample_indices(black_box(&mut rng), black_box(&weights), black_box(&mut out));
        }
        let elapsed = t0.elapsed();
        let ns_per_call = elapsed.as_nanos() as f64 / n_runs as f64;
        let ns_per_step = ns_per_call / (m + n) as f64;
        println!(
            "  {:>8}  {:>14.0}  {:>14.2}",
            m, ns_per_call, ns_per_step
        );
    }
}

/// Run a fenced microbenchmark. Returns ns/sample.
fn bench_one<F>(k: u32, n: u64, mut f: F) -> f64
where
    F: FnMut(&mut StdRng, u32) -> f64,
{
    // Fixed seed so both methods see identical PRNG sequences (modulo their
    // internal consumption rates).
    let mut rng = StdRng::seed_from_u64(0xBEEF);

    // Warmup.
    let mut s = 0.0_f64;
    for _ in 0..10_000 {
        s += f(&mut rng, k);
    }
    black_box(s);

    let t0 = Instant::now();
    let mut acc = 0.0_f64;
    for _ in 0..n {
        // Fence around both inputs and outputs of every call. This prevents
        // LLVM from vectorizing the loop body or hoisting work across calls.
        let kk = black_box(k);
        let x = f(black_box(&mut rng), kk);
        acc = black_box(acc + x);
    }
    let elapsed = t0.elapsed();
    black_box(acc);

    elapsed.as_nanos() as f64 / n as f64
}
