//! Gaussian process
//!
//! TODO here illustrate both ways of using a gaussian process

use crate::parameters::{kernel::Kernel, prior::Prior};
use crate::gaussian_process_nalgebra::GaussianProcess_nalgebra;

mod builder;
mod constructors;
mod fit;
mod predict;

// added with public visibility here for documentation purposed
pub use builder::GaussianProcessBuilder;
pub use crate::algebra::MultivariateNormal;

/// A Gaussian process that can be used to make predictions based on its training data
pub struct GaussianProcess<KernelType: Kernel, PriorType: Prior>
{
   /// gaussian process storing the information
   gp: GaussianProcess_nalgebra<KernelType, PriorType>
}
