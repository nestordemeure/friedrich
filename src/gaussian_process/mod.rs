//! Gaussian process
//!
//! TODO here illustrate both ways of using a gaussian process

use crate::parameters::{kernel::Kernel, prior::Prior};
use crate::gaussian_process_nalgebra::NAlgebraGaussianProcess;
use std::ops::{Deref, DerefMut};

mod builder;
mod constructors;
mod fit;
mod predict;

// added with public visibility here for documentation purposed
pub use builder::GaussianProcessBuilder;

/// A Gaussian process that can be used to make predictions based on its training data
pub struct GaussianProcess<KernelType: Kernel, PriorType: Prior>
{
   /// gaussian process storing the information
   gp: NAlgebraGaussianProcess<KernelType, PriorType>
}

/// automatic dereference from GaussianProcess to NAlgebraGaussianProcess
impl<KernelType: Kernel, PriorType: Prior> Deref for GaussianProcess<KernelType, PriorType>
{
   type Target = NAlgebraGaussianProcess<KernelType, PriorType>;
   fn deref(&self) -> &Self::Target
   {
      &self.gp
   }
}

/// automatic mutable dereference from GaussianProcess to NAlgebraGaussianProcess
impl<KernelType: Kernel, PriorType: Prior> DerefMut for GaussianProcess<KernelType, PriorType>
{
   fn deref_mut(&mut self) -> &mut Self::Target
   {
      &mut self.gp
   }
}
