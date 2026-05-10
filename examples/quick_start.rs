//! Runnable version of the README Quick Start.
//!
//! `cargo run --example quick_start` builds and runs this. Prints
//! a histogram of the resampled indices and a sanity-check ratio
//! against the input weights.

use ltsis::resample_indices;
use rand::SeedableRng;

fn main() {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    // Some weighted population. Weights need not be normalized.
    let weights = vec![1.0_f32, 3.0, 2.0, 4.0];

    // Resample 1000 indices from this distribution.
    let mut out = vec![0_u32; 1000];
    resample_indices(&mut rng, &weights, &mut out);

    // Tally how often each index appears.
    let mut counts = vec![0_u32; weights.len()];
    for &idx in &out {
        counts[idx as usize] += 1;
    }

    // Print results. Each index's count should be roughly proportional
    // to its weight: with weights [1, 3, 2, 4] and 1000 draws, expect
    // about [100, 300, 200, 400].
    let total_weight: f32 = weights.iter().sum();
    println!("idx  weight   p(idx)   count   p̂(count)");
    println!("---  ------   ------   -----   --------");
    for (i, (&w, &c)) in weights.iter().zip(counts.iter()).enumerate() {
        let p = w / total_weight;
        let p_hat = c as f32 / out.len() as f32;
        println!("{i:3}  {w:6.1}   {p:6.3}   {c:5}   {p_hat:6.3}");
    }

    // Output is in ascending order: out[0] ≤ out[1] ≤ … ≤ out[n-1].
    // To use these as slice indices, cast to usize:
    //
    //   let particles: Vec<MyParticle> = ...;
    //   let resampled: Vec<MyParticle> =
    //       out.iter().map(|&i| particles[i as usize].clone()).collect();
    //
    // For a particle filter, this is the "refresh the population"
    // step.
}
