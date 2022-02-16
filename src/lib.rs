//! # Friedrich : Gaussian Process Regression
//!
//! This library implements [Gaussian Process Regression](https://en.wikipedia.org/wiki/Gaussian_process) in Rust.
//! Our goal is to provide a building block for other algorithms (such as [Bayesian Optimization](https://en.wikipedia.org/wiki/Bayesian_optimization)).
//!
//! Gaussian process have both the ability to extract a lot of information from their training data and to return a prediction and an uncertainty on their prediction.
//! Furthermore, they can handle non-linear phenomenons, take uncertainty on the inputs into account and encode a prior on the output.
//!
//! All of those properties make them an algorithm of choice to perform regression when data is scarce or when having uncertainty bars on the output is a desirable property.
//!
//! However, the `o(n^3)` complexity of the algorithm makes the classical implementation unsuitable for large training datasets.
//!
//! ## Functionalities
//!
//! This implementation lets you:
//!
//! - Define a gaussian process with default parameters or using the builder pattern.
//! - Train it on multidimensional data.
//! - Fit the parameters (kernel, prior and noise) on the training data.
//! - Add additional samples efficiently (`O(n^2)`) and refit the process.
//! - Predict the mean, variance and covariance matrix for given inputs.
//! - Sample the distribution at a given position.
//! - Save and load a trained model with [serde](https://serde.rs/).
//!
//! ## Inputs
//!
//! Most methods of this library can currently work with the following `input -> output` pairs :
//!
//! Input | Output | Description
//! ---|---|---
//! [`Array2<f64>`](https://docs.rs/ndarray/0.15/ndarray/type.Array2.html) | [`Array1<f64>`](https://docs.rs/ndarray/0.15/ndarray/type.Array1.html) | Multiple input vectors to multiple output values (with `friedrich_ndarray` feature).
//! [`Array1<f64>`](https://docs.rs/ndarray/0.15/ndarray/type.Array1.html) | [`f64`] | A single input vector to a single output value (with `friedrich_ndarray` feature).
//! [`DMatrix<f64>`](https://docs.rs/nalgebra/0.29/nalgebra/base/type.DMatrix.html) | [`DVector<f64>`](https://docs.rs/nalgebra/0.29/nalgebra/base/type.DVector.html) | Multiple input vectors to multiple output values.
//! [`DVector<f64>`](https://docs.rs/nalgebra/0.29/nalgebra/base/type.DVector.html) | [`f64`] | A single input vector to a single output value.
//! [`Vec<Vec<f64>>`] | [`Vec<f64>` ] | Multiple input vectors to multiple output values.
//! [`Vec<f64>`] | [`f64` ] | A single input vector to a single input value.
//!
//! See the [`Input`] trait if you want to add you own input type.
//!
mod algebra;
mod conversion;
pub mod gaussian_process;
mod parameters;
pub use algebra::{SMatrix, SRowVector, SVector};
pub use conversion::Input;
pub use parameters::*;
