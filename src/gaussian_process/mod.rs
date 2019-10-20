//! Builds a gaussian process
//!
//! TODO here illustrate both ways of using a gaussian process

mod builder;
mod trained;

use crate::parameters::*;
pub use builder::GaussianProcessBuilder;
pub use trained::*;
pub use crate::algebra::MultivariateNormal;
pub use crate::conversion::{AsVector, AsMatrix};

/// Quick constructors for gaussian processes
pub struct GaussianProcess {}

impl GaussianProcess
{
   /// returns a builder to design a gaussian process adapted to the problem
   pub fn new<InMatrix: AsMatrix, OutVector: AsVector>(
      training_inputs: InMatrix,
      training_outputs: OutVector)
      -> GaussianProcessBuilder<kernel::Gaussian, prior::Constant, InMatrix, OutVector>
   {
      GaussianProcessBuilder::<kernel::Gaussian, prior::Constant, InMatrix, OutVector>::new(training_inputs,
                                                                                            training_outputs)
   }

   /*
   /// returns a default gaussian process with a gaussian kernel and a constant prior, both fitted to the data
   pub fn default<InMatrix: AsMatrix, OutVector: AsVector>(
      training_inputs: InMatrix,
      training_outputs: OutVector)
      -> GaussianProcessTrained<kernel::Gaussian, prior::Constant, OutVector>
   {
      GaussianProcessBuilder::<kernel::Gaussian, prior::Constant, InMatrix, OutVector>::new(training_inputs, training_outputs)
      .fit_kernel()
      .fit_prior()
      .train()
   }*/
}
