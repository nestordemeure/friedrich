# TODO

List of features that will be implemented (in no particular order) :

- Integrate nalgebra cholesky decomposition update once the new version is released
- Add [ndarray](https://docs.rs/ndarray/) support behind a feature flag
- Add [serde](https://docs.rs/ndarray/) support behind a feature flag
- Add simple [kernel regression](https://en.wikipedia.org/wiki/Kernel_regression#Nadaraya%E2%80%93Watson_kernel_regression) (not as clever but much faster).
- add [benchmarks](http://www.resibots.eu/limbo/release-2.0/reg_benchmarks.html) to confirm correctness and validate performances
- Add sparse gaussian proces
- Improve test coverage

List of API modifications that are being considered :

- replace the builder pattern with a macro (might rely on [duang](https://crates.io/crates/duang))
- simplify the kernel trait

*Do not hesitate to start an issue if you want to contribute to one of those points.*
