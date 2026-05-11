* I've corrected a bunch of stuff: start by noting my
  changes and reviewing them.

* Everywhere we talk about "resampling" we should instead be
  talking about "sampling": the resampling terminology is
  for the now-discarded BPF special case; this includes the
  naming of the primitives `resample_` → `sample`.

* Use LaTeX math in markdown (if github supports
  this) and rustdoc (if cargo doc supports this).

* Input weight length bounds should be asserted even in
  release builds. Sum of weights being positive should be
  asserted even in release builds. We should bench asserting
  weights being non-negative finite in release builds, and
  do so unless the cost is prohibitive.

* `resample_indices` should give an iterator. 

* The last paragraph of INTERNALS 4.4 should be carefully
  reread to see if it is still true: I think we can't run
  off the end in resampling anymore?

* In INTERNALS 5.5, the last paragraph about SIMD is still
  pretty confusing; in particular, we expect and want the
  block sampler to vectorize across the block on machines
  that can do this, I think?

* We should probably split the streaming and block samplers
  as separate Cargo features, so that the direct `libm`
  dependency can be removed for `no_std` in the block case
  (or some similar fix).

* The module structure is excessive here. All the library
  code should be combined as a single flat `lib.rs` file.

* The statistical tests need much more supporting
  documentation in `INTERNALS.md`: I am skeptical that even
  an expert could validate the correctness of the tests. In
  particular, even "standard" references
  (e.g. Wilson-Hilferty) should have citations.

