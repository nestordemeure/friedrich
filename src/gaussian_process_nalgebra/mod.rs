//! Raw nalgebra Gaussian process
//!
//! Used for raw acess to the underlying implementation.
//! Mostly usefull to reduce copies and provide binding for other input/output types.
//!
//! TODO here illustrate both ways of using a gaussian process

use crate::parameters::{kernel::Kernel, prior::Prior};
use crate::algebra::{EMatrix, EVector};
use nalgebra::{Cholesky, Dynamic};

mod constructors;
mod fit;
mod predict;

// added with public visibility here for documentation purposed
pub use crate::algebra::MultivariateNormal;

/// A Gaussian process that can be used to make predictions based on its training data
pub struct NAlgebraGaussianProcess<KernelType: Kernel, PriorType: Prior>
{
   /// value to which the process will regress in the absence of informations
   pub prior: PriorType,
   /// kernel used to fit the process on the data
   pub kernel: KernelType,
   /// amplitude of the noise of the data
   pub noise: f64,
   /// data used for fit
   training_inputs: EMatrix,
   training_outputs: EVector,
   /// cholesky decomposition of the covariance matrix trained on the current datapoints
   covmat_cholesky: Cholesky<f64, Dynamic>
}
