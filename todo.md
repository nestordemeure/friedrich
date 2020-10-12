# TODO

List of features that should be implemented (in no particular order) :

- Integrate nalgebra cholesky decomposition update once the new version is released
- Add [serde](https://docs.rs/ndarray/) support behind a feature flag
- Add simple [kernel regression](https://en.wikipedia.org/wiki/Kernel_regression#Nadaraya%E2%80%93Watson_kernel_regression) (not as clever but much faster).
- add [benchmarks](http://www.resibots.eu/limbo/release-2.0/reg_benchmarks.html) to confirm correctness and validate performances
- Add sparse gaussian proces
- Improve test coverage

List of algorithmic modifications that might be implemented (depending on their impact on perf/accuracy/code complexity) :

- Reduce memory usage (the fit, in particular, could use a lot less memory)
- Store the original output vector (this might simplify some formula)
- Store `cov^-1*output vector` (this would make predictions much faster once the model is trained)

List of API modifications that are being considered :

- Replace the builder pattern with a macro (might rely on [duang](https://crates.io/crates/duang) or something similar)
- Simplify the kernel trait

*Do not hesitate to start an issue if you want to contribute to one of those points.*
