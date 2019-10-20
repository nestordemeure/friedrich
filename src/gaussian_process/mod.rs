//! Gaussian process
//!
//! TODO here illustrate both ways of using a gaussian process

use crate::conversion::AsVector;
use crate::parameters::{kernel::Kernel, prior::Prior};
use crate::algebra::{EMatrix, EVector};
use std::marker::PhantomData;
use nalgebra::{Cholesky, Dynamic};

mod builder;
mod constructors;
mod fit;
mod predict;

// added with public visibility here for documentation purposed
pub use builder::GaussianProcessBuilder;
pub use crate::algebra::MultivariateNormal;
pub use crate::conversion::*;

/// A Gaussian process that can be used to make predictions based on its training data
pub struct GaussianProcess<KernelType: Kernel, PriorType: Prior, OutVector: AsVector>
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
   /// types of the inputs and outputs
   output_type: PhantomData<OutVector>,
   /// cholesky decomposition of the covariance matrix trained on the current datapoints
   covmat_cholesky: Cholesky<f64, Dynamic>
}
