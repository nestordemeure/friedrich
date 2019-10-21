//! Methods to fit a gaussian process on new data.

use nalgebra::{DVector, DMatrix};
use crate::parameters::kernel::Kernel;
use crate::parameters::prior::Prior;
use super::GaussianProcess;
use crate::algebra;

impl<KernelType: Kernel, PriorType: Prior> GaussianProcess<KernelType, PriorType>
{
   //----------------------------------------------------------------------------------------------
   // TRAINING

   /// adds new samples to the model
   /// update the model (which is faster than a training from scratch)
   /// does not refit the parameters
   pub fn add_samples_several(&mut self, inputs: &[Vec<f64>], outputs: &[f64])
   {
      // converts input to correct format
      let inputs = algebra::make_matrix_from_row_slices(inputs);
      let outputs = DVector::from_column_slice(outputs);
      // add samples
      self.gp.add_samples(&inputs, &outputs)
   }

   /// adds new sample to the model
   /// update the model (which is faster than a training from scratch)
   /// does not refit the parameters
   pub fn add_sample(&mut self, input: &[f64], output: f64)
   {
      // converts input to correct format
      let input = DMatrix::from_row_slice(1, input.len(), input);
      let output = DVector::from_element(1, output);
      // add samples
      self.gp.add_samples(&input, &output)
   }

   /// fits the parameters if requested and retrain the model from scratch if needed
   pub fn fit_parameters(&mut self, fit_prior: bool, fit_kernel: bool)
   {
      self.gp.fit_parameters(fit_prior, fit_kernel)
   }

   /// adds new samples to the model and fit the parameters
   /// faster than doing add_samples().fit_parameters()
   pub fn add_samples_fit_several(&mut self,
                                  inputs: &[Vec<f64>],
                                  outputs: &[f64],
                                  fit_prior: bool,
                                  fit_kernel: bool)
   {
      // converts input to correct format
      let inputs = algebra::make_matrix_from_row_slices(inputs);
      let outputs = DVector::from_column_slice(outputs);
      // add samples
      self.gp.add_samples_fit(&inputs, &outputs, fit_prior, fit_kernel);
   }

   /// adds new sample to the model and fit the parameters
   /// faster than doing add_samples().fit_parameters()
   pub fn add_sample_fit(&mut self, input: &[f64], output: f64, fit_prior: bool, fit_kernel: bool)
   {
      // converts input to correct format
      let input = DMatrix::from_row_slice(1, input.len(), input);
      let output = DVector::from_element(1, output);
      // add samples
      self.gp.add_samples_fit(&input, &output, fit_prior, fit_kernel)
   }
}
