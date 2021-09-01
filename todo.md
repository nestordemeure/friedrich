# TODO

List of features that should be implemented (in no particular order):

- Add forced cholesky decomposition (once the PR passes in nalgebra) to eliminate Cholesky failures (update doc and readme accordingly)
- Add simple [kernel regression](https://en.wikipedia.org/wiki/Kernel_regression#Nadaraya%E2%80%93Watson_kernel_regression) (not as clever but much faster).

List of algorithmic modifications that might be implemented (depending on their impact on perf/accuracy/code complexity):

- Reduce memory usage (the fit, in particular, could use a lot less memory)
- Store the original output vector (this might simplify some formula)
- Store `cov^-1*output vector` (this would make predictions much faster once the model is trained)

List of API modifications that are being considered:

- Replace the builder pattern with a macro (might rely on [duang](https://crates.io/crates/duang) or something similar)
- Simplify the kernel trait

List of infrastructure improvements that are being considered:

- add [benchmarks](http://www.resibots.eu/limbo/release-2.0/reg_benchmarks.html) to confirm correctness and validate performances
- Improve test coverage

*Do not hesitate to start an issue if you want to contribute to one of those points.*
