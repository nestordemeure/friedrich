//! # Friedrich : Gaussian Process Regression
//!
//! This libarie implements [Gaussian Process Regression](https://en.wikipedia.org/wiki/Gaussian_process) in Rust.
//! Our goal is to provide a building block for other algorithms (such as [Bayesian Optimization](https://en.wikipedia.org/wiki/Bayesian_optimization)).
//!
//! Gaussian process have both the ability to extract a lot of information from their training data and to return a prediction and an uncertainty on their prediction.
//! Furthermore, they can handle non-linear phenomenons, take uncertainty on the inputs into account and encode a prior on the output.
//!
//! All of those properties make them an algorithm of choice to perform regression when data is scarce or when having uncertainty bars on the ouput is a desirable property.
//!
//! However, the `o(n^3)` complexity of the algorithm makes the classical implementation unsuitable for large training datasets.
//!
//! ## Functionalities
//!
//! This implementation lets you :
//!
//! - define a gaussian process with default parameters or using the builder pattern
//! - train it on multidimensional data
//! - fit the parameters (kernel, prior and noise) on the training data
//! - add additional samples and refit the process
//! - predict the mean and variance and covariance matrix for given inputs
//! - sample the distribution at a given position
//!
//! ## Inputs
//!
//! Most methods of this library can currently work with the following `input -> ouput` pairs :
//!
//! - `Vec<f64> -> f64` a single, multidimensional, sample
//! - `Vec<Vec<f64>> -> Vec<f64>` each inner vector is a training sample
//! - `DMatrix<f64> -> DVector<f64>` using a [nalgebra](https://www.nalgebra.org/) matrix with one row per sample
//! - `ArrayBase<f64, Ix1> -> f64` a single sample stored in a [ndarray](https://crates.io/crates/ndarray) array (using the `friedrich_ndarray` feature)
//! - `ArrayBase<f64, Ix2> -> Array1<f64>` each row is a sample (using the `friedrich_ndarray` feature)
//!
//! See the `Input` trait if you want to add you own input type.
//!
mod algebra;
mod conversion;
pub mod gaussian_process;
mod parameters;
pub use conversion::Input;
pub use parameters::*;
