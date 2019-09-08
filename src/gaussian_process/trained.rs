//! Trained Gaussian process

use nalgebra::{DMatrix, Cholesky, Dynamic};
use crate::parameters::kernel::Kernel;
use crate::parameters::prior::Prior;
use crate::matrix;

/// gaussian process
pub struct GaussianProcessTrained<KernelType: Kernel, PriorType: Prior>
{
   /// value to which the process will regress in the absence of informations
   prior: PriorType,
   /// kernel used to fit the process on the data
   kernel: KernelType,
   /// amplitude of the noise of the data
   noise: f64,
   /// data used for fit
   training_inputs: DMatrix<f64>,
   training_outputs: DMatrix<f64>,
   /// cholesky decomposition of the covariance matrix trained on the current datapoints
   covmat_cholesky: Cholesky<f64, Dynamic>
}

impl<KernelType: Kernel, PriorType: Prior> GaussianProcessTrained<KernelType, PriorType>
{
   pub fn new(prior: PriorType,
              kernel: KernelType,
              noise: f64,
              training_inputs: DMatrix<f64>,
              training_outputs: DMatrix<f64>)
              -> Self
   {
      let training_outputs = training_outputs - prior.prior(&training_inputs);
      let covmat_cholesky = matrix::make_covariance_matrix(&training_inputs, &training_inputs, &kernel)
                  .cholesky().expect("Cholesky decomposition failed!");
      GaussianProcessTrained::<KernelType, PriorType> { prior,
                                                        kernel,
                                                        noise,
                                                        training_inputs,
                                                        training_outputs,
                                                        covmat_cholesky }
   }

   //----------------------------------------------------------------------------------------------
   // TRAINING

   /// adds new samples to the model
   /// update the model (which is faster than a training from scratch)
   /// does not refit the parameters
   pub fn add_samples(&mut self, inputs: DMatrix<f64>, outputs: DMatrix<f64>)
   {
      // growths the training matrix
      matrix::add_rows(&mut self.training_inputs, &inputs);
      matrix::add_rows(&mut self.training_outputs, &outputs);

      // recompute cholesky matrix
      self.covmat_cholesky = matrix::make_covariance_matrix(&self.training_inputs, &self.training_inputs, &self.kernel)
                                                .cholesky().expect("Cholesky decomposition failed!");
      // TODO update cholesky matrix instead of recomputing it from scratch
   }

   /// fits the parameters and retrain the model from scratch
   pub fn fit_parameters(&mut self, fit_prior: bool, fit_kernel: bool)
   {
      if fit_prior
      {
         // gets the original data back in order to update the prior
         let original_training_outputs = &self.training_outputs + self.prior.prior(&self.training_inputs);
         self.prior.fit(&self.training_inputs, &original_training_outputs);
         self.training_outputs = original_training_outputs - self.prior.prior(&self.training_inputs);
         // NOTE: adding and substracting each time we fit a prior might be numerically unstable
      }

      if fit_kernel
      {
         // fit kernel using new data and new prior
         self.kernel.fit(&self.training_inputs, &self.training_outputs);
      }

      self.covmat_cholesky = matrix::make_covariance_matrix(&self.training_inputs, &self.training_inputs, &self.kernel)
                                                .cholesky().expect("Cholesky decomposition failed!");
   }

   /// adds new samples to the model and fit the parameters
   /// faster than doing add_samples().fit_parameters()
   pub fn add_samples_fit(&mut self,
                          inputs: DMatrix<f64>,
                          outputs: DMatrix<f64>,
                          fit_prior: bool,
                          fit_kernel: bool)
   {
      // growths the training matrix
      matrix::add_rows(&mut self.training_inputs, &inputs);
      matrix::add_rows(&mut self.training_outputs, &outputs);

      // refit the parameters and retrain the model from scratch
      self.fit_parameters(fit_prior, fit_kernel);
   }

   //----------------------------------------------------------------------------------------------
   // PREDICTION

   /// computes a prediction per row of the input matrix
   pub fn predict_mean(&mut self, inputs: DMatrix<f64>) -> DMatrix<f64>
   {
      // computes weights to give each training sample
      let input_cov = matrix::make_covariance_matrix(&self.training_inputs, &inputs, &self.kernel);
      let weights = self.covmat_cholesky.solve(&input_cov);

      // computes prior for the given inputs
      let mut prior = self.prior.prior(&inputs);

      //weights.transpose() * &self.training_outputs + prior
      prior.gemm_tr(1f64, &weights, &self.training_outputs, 1f64);
      prior
   }
}
