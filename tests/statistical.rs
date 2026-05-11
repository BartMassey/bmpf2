//! Integration tests for `ltsis`.
//!
//! Each test exercises one invariant statistically with a calibrated
//! threshold. Per-test random-failure probability under correct code
//! is < 1e-10; aggregate across all tests is < 1e-9 (the user-stated
//! ceiling). RNG seeds are fixed for reproducibility within a given
//! `rand` version.
//!
//! Methodology is documented in `INTERNALS.md` §5.4.

use ltsis::{first_uniform, sample_indices, sample_indices_buffered, SortedUniforms};
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};

// ---------------------------------------------------------------------------
// Calibrated thresholds for <1e-10 per-test false-failure probability.
// ---------------------------------------------------------------------------

/// One- and two-sample Kolmogorov–Smirnov critical-value coefficient.
/// Asymptotic two-sided KS: P(D > c/√n) ≈ 2·exp(−2c²). At c = 3.7 this
/// is ≈ 2.5e-12 per sub-check, well below 1e-10 even with up to 7
/// sub-checks per test.
const KS_CRITICAL: f64 = 3.7;

/// Standard-normal upper-tail z for sub-checks where one comparison
/// per parameter setting is made (Test_moments mean checks). At
/// z = 7.0, P(|Z| > 7.0) ≈ 2.6e-12.
const NORMAL_Z_SINGLE: f64 = 7.0;

/// Standard-normal z for max-over-positions tests (sorted-uniforms
/// per-position moments, up to 100 positions per test). At z = 7.5,
/// P(|Z| > 7.5) ≈ 6.4e-14, so 100·P ≈ 6.4e-12.
const NORMAL_Z_MAX_OVER_POS: f64 = 7.5;

/// Wilson–Hilferty z parameter for chi-squared upper-tail probability.
/// At z = 7.0 the corresponding upper tail is ≈ 1.3e-12 per sub-case.
const CHISQ_Z: f64 = 7.0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Naive O(k) oracle: literally draw `k` uniforms and return the
/// minimum. Used as an independent reference distribution that goes
/// through none of the library's machinery.
fn min_of_k_uniforms_naive<R: Rng + ?Sized>(rng: &mut R, k: u32) -> f32 {
    let mut m: f32 = 1.0;
    for _ in 0..k {
        let u: f32 = rng.random();
        if u < m {
            m = u;
        }
    }
    m
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

/// Construct an `f32` weight vector from f64 literals.
fn weights_from_f64(ws: &[f64]) -> Vec<f32> {
    ws.iter().map(|&w| w as f32).collect()
}

/// Naive multinomial sampler (f64-internal): for each output, draw
/// a uniform on [0, total) and binary-search the cumulative-weight
/// array. O(m + n log m). The trusted reference for the chi-squared
/// sampler-vs-multinomial comparison.
fn naive_multinomial<R: Rng + ?Sized>(rng: &mut R, weights: &[f32], out: &mut [u32]) {
    let mut cum = vec![0.0_f64; weights.len()];
    let mut t = 0.0_f64;
    for (i, &w) in weights.iter().enumerate() {
        t += w as f64;
        cum[i] = t;
    }
    for slot in out.iter_mut() {
        let u: f64 = rng.random();
        let target = u * t;
        let idx = cum.partition_point(|&c| c <= target);
        *slot = idx.min(weights.len() - 1) as u32;
    }
}

/// Wilson–Hilferty approximation of the chi-squared upper-tail
/// critical value at the given dof and z (standard-normal quantile).
fn chisq_critical(dof: i32, z: f64) -> f64 {
    let dofs = dof as f64;
    dofs * (1.0 - 2.0 / (9.0 * dofs) + z * (2.0 / (9.0 * dofs)).sqrt()).powi(3)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Deterministic range check: every `first_uniform` sample lies in
/// `[0, 1)`. No false-failure probability under correct code.
#[test]
fn range() {
    let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
    for &k in &[1u32, 2, 3, 5, 10, 100, 1000] {
        for _ in 0..100_000 {
            let x = first_uniform(&mut rng, k);
            assert!(
                !x.is_nan() && (0.0..1.0).contains(&x),
                "k={k}: out-of-range {x:+e}"
            );
        }
    }
}

/// Empirical first / second moments of `first_uniform` match the
/// closed-form `Beta(1, k)` values.
///
/// For X ~ Beta(1, k):  E[X] = 1/(k+1),  Var[X] = k / [(k+1)²(k+2)].
///
/// Mean check tolerance: NORMAL_Z_SINGLE σ.
/// Variance check tolerance: 2% relative (≈ 14σ at n=10⁶ samples,
/// far below 1e-10).
#[test]
fn moments_first_uniform() {
    let mut rng = SmallRng::seed_from_u64(0xDEADBEEF);
    let n_samples = 1_000_000;

    for &k in &[1u32, 2, 3, 5, 10, 50, 200] {
        let kf = k as f64;
        let theo_mean = 1.0 / (kf + 1.0);
        let theo_var = kf / ((kf + 1.0).powi(2) * (kf + 2.0));

        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        for _ in 0..n_samples {
            let x = first_uniform(&mut rng, k) as f64;
            sum += x;
            sum_sq += x * x;
        }
        let mean = sum / n_samples as f64;
        let var = sum_sq / n_samples as f64 - mean * mean;

        let se_mean = (theo_var / n_samples as f64).sqrt();
        let z = (mean - theo_mean).abs() / se_mean;
        let var_rel = (var - theo_var).abs() / theo_var;

        assert!(
            z < NORMAL_Z_SINGLE,
            "k={k}: mean = {mean:.6} (theo {theo_mean:.6}, z = {z:.2}σ, threshold {NORMAL_Z_SINGLE}σ)"
        );
        assert!(
            var_rel < 0.02,
            "k={k}: var rel-err = {var_rel:.2e} > 0.02 (var = {var:.6}, theo {theo_var:.6})"
        );
    }
}

/// One-sample Kolmogorov–Smirnov of `first_uniform` against the
/// analytic CDF of Beta(1, k):  F_k(x) = 1 − (1 − x)^k.
#[test]
fn ks_against_theory() {
    let mut rng = SmallRng::seed_from_u64(0x12345678);
    let n = 50_000;

    for &k in &[1u32, 2, 3, 5, 10, 50, 200] {
        let mut samples: Vec<f64> = (0..n).map(|_| first_uniform(&mut rng, k) as f64).collect();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let kf = k as f64;
        let mut d_max: f64 = 0.0;
        for (i, &x) in samples.iter().enumerate() {
            let f_emp_above = (i + 1) as f64 / n as f64;
            let f_emp_below = i as f64 / n as f64;
            let f_theory = 1.0 - (1.0 - x).powf(kf);
            let d = (f_emp_above - f_theory)
                .abs()
                .max((f_theory - f_emp_below).abs());
            if d > d_max {
                d_max = d;
            }
        }

        let critical = KS_CRITICAL / (n as f64).sqrt();
        assert!(
            d_max < critical,
            "k={k}: KS D = {d_max:.5} ≥ critical {critical:.5}"
        );
    }
}

/// Two-sample Kolmogorov–Smirnov: `first_uniform` vs. an independent
/// min-of-k-uniforms oracle. The oracle exercises none of the
/// library's machinery, so this is a cross-check that `first_uniform`
/// matches the distribution it claims to sample.
#[test]
fn ks_against_min_oracle() {
    let mut rng_a = SmallRng::seed_from_u64(0xAAAA_AAAA);
    let mut rng_b = SmallRng::seed_from_u64(0xBBBB_BBBB);
    let n = 20_000;

    for &k in &[2u32, 3, 5, 10, 50] {
        let mut a: Vec<f64> = (0..n)
            .map(|_| first_uniform(&mut rng_a, k) as f64)
            .collect();
        let mut b: Vec<f64> = (0..n)
            .map(|_| min_of_k_uniforms_naive(&mut rng_b, k) as f64)
            .collect();
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        b.sort_by(|x, y| x.partial_cmp(y).unwrap());

        let d_max = two_sample_ks(&a, &b);
        // Two-sample KS critical: c · √((n_a + n_b) / (n_a · n_b)).
        let critical = KS_CRITICAL * ((2 * n) as f64 / (n * n) as f64).sqrt();
        assert!(
            d_max < critical,
            "k={k}: two-sample KS D = {d_max:.5} ≥ critical {critical:.5}"
        );
    }
}

/// Empirical mean and variance of each order-statistic position of
/// `SortedUniforms` match the closed-form values.
///
/// For the i-th of n sorted uniforms (1-indexed):
///   E[U_(i)]   = i / (n + 1)
///   Var[U_(i)] = i (n − i + 1) / [(n + 1)² (n + 2)]
///
/// Mean check is bounded over all n_positions; tolerance set so
/// `n_positions · P(|Z| > NORMAL_Z_MAX_OVER_POS) < 1e-10`.
#[test]
fn sorted_uniforms_moments() {
    let mut rng = SmallRng::seed_from_u64(0xABCD_1234);

    for &n in &[5u32, 20, 100] {
        let n_runs = 200_000;

        let mut sum = vec![0.0_f64; n as usize];
        let mut sum_sq = vec![0.0_f64; n as usize];

        for _ in 0..n_runs {
            let mut iter = SortedUniforms::new(&mut rng, n);
            for i in 0..n as usize {
                let v = iter.next().unwrap() as f64;
                sum[i] += v;
                sum_sq[i] += v * v;
            }
        }

        for i in 0..n as usize {
            let i1 = (i + 1) as f64;
            let nf = n as f64;
            let theo_mean = i1 / (nf + 1.0);
            let theo_var = i1 * (nf - i1 + 1.0) / ((nf + 1.0).powi(2) * (nf + 2.0));

            let emp_mean = sum[i] / n_runs as f64;
            let emp_var = sum_sq[i] / n_runs as f64 - emp_mean * emp_mean;

            let se_mean = (theo_var / n_runs as f64).sqrt();
            let z = (emp_mean - theo_mean).abs() / se_mean;
            let var_rel = (emp_var - theo_var).abs() / theo_var;

            assert!(
                z < NORMAL_Z_MAX_OVER_POS,
                "n={n} pos={pos}: mean z = {z:.2}σ, threshold {NORMAL_Z_MAX_OVER_POS}σ \
                 (emp_mean={emp_mean:.6}, theo_mean={theo_mean:.6})",
                pos = i + 1
            );
            assert!(
                var_rel < 0.05,
                "n={n} pos={pos}: var rel-err = {var_rel:.2e} > 0.05",
                pos = i + 1
            );
        }
    }
}

/// Pooled KS test: the marginal distribution of all yielded sorted
/// uniforms across many runs is Uniform(0, 1).
#[test]
fn sorted_uniforms_pooled_ks() {
    let mut rng = SmallRng::seed_from_u64(0xCAFE_F00D);

    for &n in &[10u32, 100, 1000] {
        let n_runs = 50_000 / n.max(1) as usize;
        let total = n_runs * n as usize;
        let mut pooled: Vec<f64> = Vec::with_capacity(total);
        for _ in 0..n_runs {
            let iter = SortedUniforms::new(&mut rng, n);
            pooled.extend(iter.map(|x| x as f64));
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
        let critical = KS_CRITICAL / len.sqrt();
        assert!(
            d_max < critical,
            "n={n} pooled={total}: KS D = {d_max:.5} ≥ critical {critical:.5}"
        );
    }
}

// ---------------------------------------------------------------------------
// Sampler tests
// ---------------------------------------------------------------------------

fn sample_marginals<R>(method: &str, mut sample_fn: R)
where
    R: FnMut(&mut SmallRng, &[f32], &mut [u32]),
{
    let mut rng = SmallRng::seed_from_u64(0xFEED_BEEF);

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
        let weights = weights_from_f64(weights_f64);
        let m = weights.len();
        let total: f64 = weights_f64.iter().sum();
        let n_per_run = 50;
        let n_runs = 4_000;
        let total_draws = n_per_run * n_runs;

        let mut counts = vec![0u64; m];
        let mut buf = vec![0u32; n_per_run];
        for _ in 0..n_runs {
            sample_fn(&mut rng, &weights, &mut buf);
            for &idx in &buf {
                counts[idx as usize] += 1;
            }
        }

        let mut chi_sq = 0.0;
        let mut dof = 0_i32;
        for i in 0..m {
            let expected = total_draws as f64 * weights_f64[i] / total;
            if expected == 0.0 {
                assert!(
                    counts[i] == 0,
                    "[{method}/{name}] zero-weight index {i} got {} counts",
                    counts[i]
                );
                continue;
            }
            let diff = counts[i] as f64 - expected;
            chi_sq += diff * diff / expected;
            dof += 1;
        }
        dof -= 1;

        let crit = chisq_critical(dof, CHISQ_Z);
        assert!(
            chi_sq < crit,
            "[{method}/{name}] χ² = {chi_sq:.2} ≥ critical {crit:.2} (dof = {dof})"
        );
    }
}

#[test]
fn sample_marginals_streaming() {
    sample_marginals("streaming", sample_indices);
}

#[test]
fn sample_marginals_buffered() {
    sample_marginals("buffered", sample_indices_buffered);
}

fn sample_vs_multinomial<R>(method: &str, mut sample_fn: R)
where
    R: FnMut(&mut SmallRng, &[f32], &mut [u32]),
{
    let mut rng_a = SmallRng::seed_from_u64(0xA1A1_A1A1);
    let mut rng_b = SmallRng::seed_from_u64(0xB2B2_B2B2);

    let test_cases: Vec<(&str, Vec<f64>)> = vec![
        ("uniform-10", vec![1.0; 10]),
        ("skewed-8", (1..=8).map(|x| x as f64).collect()),
        (
            "peaky-15",
            (0..15).map(|i| if i == 7 { 30.0 } else { 1.0 }).collect(),
        ),
    ];

    for (name, weights_f64) in &test_cases {
        let weights = weights_from_f64(weights_f64);
        let m = weights.len();
        let n_per_run = 100;
        let n_runs = 2_000;

        let mut counts_a = vec![0u64; m];
        let mut counts_b = vec![0u64; m];
        let mut buf = vec![0u32; n_per_run];

        for _ in 0..n_runs {
            sample_fn(&mut rng_a, &weights, &mut buf);
            for &idx in &buf {
                counts_a[idx as usize] += 1;
            }
            naive_multinomial(&mut rng_b, &weights, &mut buf);
            for &idx in &buf {
                counts_b[idx as usize] += 1;
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

        let crit = chisq_critical(dof, CHISQ_Z);
        assert!(
            chi_sq < crit,
            "[{method}/{name}] two-sample χ² = {chi_sq:.2} ≥ critical {crit:.2} (dof = {dof})"
        );
    }
}

#[test]
fn sample_vs_multinomial_streaming() {
    sample_vs_multinomial("streaming", sample_indices);
}

#[test]
fn sample_vs_multinomial_buffered() {
    sample_vs_multinomial("buffered", sample_indices_buffered);
}
