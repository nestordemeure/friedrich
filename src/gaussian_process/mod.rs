pub mod builder;
pub mod trained;

use crate::input::{Input, Output};
use crate::parameters::*;
use builder::GaussianProcessBuilder;
use trained::GaussianProcessTrained;

/// struct used to define a gaussian process
pub struct GaussianProcess {}

impl GaussianProcess
{
   /// returns a builder to design a gaussian process adapted to the problem
   pub fn new<InMatrix: Input, OutVector: Output>(
      training_inputs: InMatrix,
      training_outputs: OutVector)
      -> GaussianProcessBuilder<kernel::Gaussian, prior::Constant>
   {
      GaussianProcessBuilder::<kernel::Gaussian, prior::Constant>::new(training_inputs, training_outputs)
   }

   /// returns a default gaussian process with a gaussian kernel and a constant prior, both fitted to the data
   pub fn default<InMatrix: Input, OutVector: Output>(
      training_inputs: InMatrix,
      training_outputs: OutVector)
      -> GaussianProcessTrained<kernel::Gaussian, prior::Constant>
   {
      GaussianProcessBuilder::<kernel::Gaussian, prior::Constant>::new(training_inputs, training_outputs)
      .fit_kernel()
      .fit_prior()
      .train()
   }
}
