//! Methods to fit a gaussian process on new data.

use nalgebra::{DVector, DMatrix};
use crate::parameters::kernel::Kernel;
use crate::parameters::prior::Prior;
use crate::algebra;
use super::GaussianProcess_nalgebra;

impl<KernelType: Kernel, PriorType: Prior>
   GaussianProcess_nalgebra<KernelType, PriorType>
{
   //----------------------------------------------------------------------------------------------
   // TRAINING

   /// adds new samples to the model
   /// update the model (which is faster than a training from scratch)
   /// does not refit the parameters
   pub fn add_samples_several(&mut self,
                                                                        inputs: &DMatrix<f64>,
                                                                        outputs: &DVector<f64>)
   {
      assert_eq!(inputs.nrows(), outputs.nrows());
      assert_eq!(inputs.ncols(), self.training_inputs.as_matrix().ncols());
      // grows the training matrix
      let outputs = outputs - self.prior.prior(&inputs);
      self.training_inputs.add_rows(&inputs);
      self.training_outputs.add_rows(&outputs);
      // recompute cholesky matrix
      self.covmat_cholesky = algebra::make_cholesky_covariance_matrix(&self.training_inputs.as_matrix(),
                                                                      &self.kernel,
                                                                      self.noise);
      // TODO update cholesky matrix instead of recomputing it from scratch
   }

   /// adds new sample to the model
   /// update the model (which is faster than a training from scratch)
   /// does not refit the parameters
   pub fn add_sample(&mut self, input: &DVector<f64>, output: f64)
   {
      let input = DMatrix::from_row_slice(1, input.nrows(), input.as_slice());
      let output = DVector::from_element(1, output);
      self.add_samples_several(&input, &output)
   }

   /// fits the parameters if requested and retrain the model from scratch if needed
   pub fn fit_parameters(&mut self, fit_prior: bool, fit_kernel: bool)
   {
      if fit_prior
      {
         // gets the original data back in order to update the prior
         let training_outputs =
            self.training_outputs.as_vector() + self.prior.prior(&self.training_inputs.as_matrix());
         self.prior.fit(&self.training_inputs.as_matrix(), &training_outputs);
         let training_outputs = training_outputs - self.prior.prior(&self.training_inputs.as_matrix());
         self.training_outputs.assign(&training_outputs);
         // NOTE: adding and substracting each time we fit a prior might be numerically unstable
      }

      if fit_kernel
      {
         // fit kernel using new data and new prior
         self.kernel.fit(&self.training_inputs.as_matrix(), &self.training_outputs.as_vector());
      }

      if fit_prior || fit_kernel
      {
         // retranis model if a fit happened
         self.covmat_cholesky = algebra::make_cholesky_covariance_matrix(&self.training_inputs.as_matrix(),
                                                                         &self.kernel,
                                                                         self.noise);
      }
   }

   /// adds new samples to the model and fit the parameters
   /// faster than doing add_samples().fit_parameters()
   pub fn add_samples_fit_several(&mut self,
                                                                            inputs: &DMatrix<f64>,
                                                                            outputs: &DVector<f64>,
                                                                            fit_prior: bool,
                                                                            fit_kernel: bool)
   {
      assert_eq!(inputs.nrows(), outputs.nrows());
      assert_eq!(inputs.ncols(), self.training_inputs.as_matrix().ncols());
      // grows the training matrix
      let outputs = outputs - self.prior.prior(&inputs);
      self.training_inputs.add_rows(&inputs);
      self.training_outputs.add_rows(&outputs);
      // refit the parameters and retrain the model from scratch
      if fit_kernel || fit_prior
      {
         self.fit_parameters(fit_prior, fit_kernel);
      }
      else
      {
         // retrains the model anyway if no fit happened
         self.covmat_cholesky = algebra::make_cholesky_covariance_matrix(&self.training_inputs.as_matrix(),
                                                                         &self.kernel,
                                                                         self.noise);
      }
   }

   /// adds new sample to the model and fit the parameters
   /// faster than doing add_samples().fit_parameters()
   pub fn add_sample_fit(&mut self,
                                             input: &DVector<f64>,
                                             output: f64,
                                             fit_prior: bool,
                                             fit_kernel: bool)
   {
      let input = DMatrix::from_row_slice(1, input.nrows(), input.as_slice());
      let output = DVector::from_element(1, output);
      self.add_samples_fit_several(&input, &output, fit_prior, fit_kernel)
   }
}
