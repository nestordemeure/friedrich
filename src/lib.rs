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
//! - fit the parameters (kernel and prior) on the training data (the fit is currently based on fast heuristics)
//! - add additional samples and refit the process
//! - predict the mean and variance and covariance matrix for given inputs
//! - sample the distribution at a given position
//!
//! ## Inputs
//!
//! Most methods of this library can currently work with the following `input -> ouput` pairs :
//!
//! - `Vec<Vec<f64>> -> Vec<f64>` each inner vector is a multidimentional training sample
//! - `Vec<f64> -> f64` a single multidimensional sample
//! - `DMatrix<f64> -> DVector<f64>` using a [nalgebra](https://www.nalgebra.org/) matrix with one row per sample
//!
//! See the `Input` trait if you want to add you own input type.
//!
mod algebra;
mod parameters;
mod conversion;
mod optimizer;
pub mod gaussian_process;
pub use parameters::*;
pub use conversion::Input;
