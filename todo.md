# TODO

List of features and modification considered for inclusion (from most likely to least likely):

- Add a timeout to the `fit` function (default to one hour), to satisfy the existing issue on the subject.
- Add forced cholesky decomposition (once the [PR passes in nalgebra](https://github.com/dimforge/nalgebra/pull/979)) to eliminate Cholesky failures (update doc and readme accordingly)

- Replace the builder pattern with a macro (might rely on [duang](https://crates.io/crates/duang) or something similar)
- Improve test coverage

- Reduce memory usage (the fit, in particular, could use a lot less memory)
- Store the original output vector (this might simplify some formula)
- Store `cov^-1*output vector` (this would make predictions much faster once the model is trained)
- Simplify the kernel trait

- add [benchmarks](http://www.resibots.eu/limbo/release-2.0/reg_benchmarks.html) to confirm correctness and validate performances
- Add simple [kernel regression](https://en.wikipedia.org/wiki/Kernel_regression#Nadaraya%E2%80%93Watson_kernel_regression) (not as clever but much faster and can be turned into a solid method with proper output uncertainty).

- Deal with multidimensional outputs (one can build one GP per output but it is much more efficient and actually meaningful to use one kernel and just rescale the output variance differantly for each output). This would require some API change to keep things working and covariance on multidimensional output will require a bit of thinking.
- Introduce low rank gaussian process using [this](https://arxiv.org/abs/1505.06195) low rank Cholesky decomposition? (it is an interesting alternative to sparse and variational GP) It might be interesting to try and couple it with an implicit matrix representation (having a fonction calling the kernel whenever a point is needed rather than building the `n²` elements matrix)

*Do not hesitate to start an issue if you want to contribute to one of those points.*