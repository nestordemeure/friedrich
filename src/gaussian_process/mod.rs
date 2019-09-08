pub mod builder;
pub mod trained;

use nalgebra::{DMatrix};
use crate::parameters::*;
use builder::GaussianProcessBuilder;
use trained::GaussianProcessTrained;

/// struct used to define a gaussian process
pub struct GaussianProcess {}

impl GaussianProcess
{
   /// returns a builder to design a gaussian process adapted to the problem
   pub fn new(training_inputs: DMatrix<f64>,
              training_outputs: DMatrix<f64>)
              -> GaussianProcessBuilder<kernel::Gaussian, prior::Constant>
   {
      GaussianProcessBuilder::<kernel::Gaussian, prior::Constant>::new(training_inputs, training_outputs)
   }

   /// returns a default gaussian process with a gaussian kernel and a constant prior, both fitted to the data
   pub fn default(training_inputs: DMatrix<f64>,
                  training_outputs: DMatrix<f64>)
                  -> GaussianProcessTrained<kernel::Gaussian, prior::Constant>
   {
      GaussianProcessBuilder::<kernel::Gaussian, prior::Constant>::new(training_inputs, training_outputs)
      .fit_kernel()
      .fit_prior()
      .train()
   }
}
