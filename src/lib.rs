//! Gaussian process
//!
//! ```rust
//! # use friedrich::gaussian_process::GaussianProcess;
//! # fn main() {
//! // trains a gaussian process on a dataset
//! let training_inputs = vec![vec![0.8], vec![1.2], vec![3.8], vec![4.2]];
//! let training_outputs = vec![3.0, 4.0, -2.0, -2.0];
//! let gp = GaussianProcess::default(training_inputs, training_outputs);
//!
//! // predicts the mean and variance of a point
//! let input = vec![1.];
//! let mean = gp.predict(&input);
//! let var = gp.predict_variance(&input);
//! println!("prediction: {} ± {}", mean, var.sqrt());
//!
//! // makes several prediction
//! let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
//! let outputs = gp.predict(&inputs);
//! println!("predictions: {:?}", outputs);
//!
//! // samples from the distribution
//! let new_inputs = vec![vec![1.0], vec![2.0]];
//! let sampler = gp.sample_at(&new_inputs);
//! let mut rng = rand::thread_rng();
//! println!("samples: {:?}", sampler.sample(&mut rng));
//! # }
//! ```

mod algebra;
mod parameters;
mod conversion;
pub mod gaussian_process;
pub use parameters::*;
