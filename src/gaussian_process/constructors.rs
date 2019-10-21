//! Methods to build a Gaussian process.

use super::GaussianProcess;
use super::builder::GaussianProcessBuilder;
use crate::parameters::{kernel::Kernel, prior::Prior};
use crate::parameters::*;
use crate::gaussian_process_nalgebra::GaussianProcess_nalgebra;
use nalgebra::{DVector, DMatrix};

impl<KernelType: Kernel, PriorType: Prior> GaussianProcess<KernelType, PriorType>
{
   pub fn new(prior: PriorType, kernel: KernelType, noise: f64, training_inputs: &[Vec<f64>], training_outputs: &[f64]) -> Self
   {
      let nb_rows = training_inputs.len();
      assert_ne!(nb_rows, 0);
      let nb_cols = training_inputs[0].len();
      let training_inputs = DMatrix::from_fn(nb_rows, nb_cols, |r,c| training_inputs[r][c] );
      let training_outputs = DVector::from_column_slice(training_outputs);
      let gp = GaussianProcess_nalgebra::new(prior, kernel, noise, training_inputs, training_outputs);
      GaussianProcess { gp }
   }
}

impl GaussianProcess<kernel::Gaussian, prior::Constant>
{
   /// returns a default gaussian process with a gaussian kernel and a constant prior, both fitted to the data
   pub fn default(training_inputs: &[Vec<f64>], training_outputs: &[f64]) -> Self
   {
      GaussianProcessBuilder::<kernel::Gaussian, prior::Constant>::new(training_inputs, training_outputs)
      .fit_kernel()
      .fit_prior()
      .train()
   }

   /// returns a default gaussian process with a gaussian kernel and a constant prior, both fitted to the data
   pub fn builder(
      training_inputs: &[Vec<f64>],
      training_outputs: &[f64])
      -> GaussianProcessBuilder<kernel::Gaussian, prior::Constant>
   {
      GaussianProcessBuilder::<kernel::Gaussian, prior::Constant>::new(training_inputs, training_outputs)
   }
}
