//! Trained Gaussian process

use nalgebra::{DVector, DMatrix, Cholesky, Dynamic};
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
      let covmat_cholesky = matrix::make_covariance_matrix(&training_inputs,
                                                           &training_inputs,
                                                           &kernel,
                                                           noise).cholesky()
                                                                 .expect("Cholesky decomposition failed!");
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
      self.covmat_cholesky =
         matrix::make_covariance_matrix(&self.training_inputs,
                                        &self.training_inputs,
                                        &self.kernel,
                                        self.noise).cholesky()
                                                   .expect("Cholesky decomposition failed!");
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

      self.covmat_cholesky =
         matrix::make_covariance_matrix(&self.training_inputs,
                                        &self.training_inputs,
                                        &self.kernel,
                                        self.noise).cholesky()
                                                   .expect("Cholesky decomposition failed!");
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

   /// predicts the mean of the gaussian process at each row of the input
   pub fn predict_mean(&self, inputs: &DMatrix<f64>) -> DMatrix<f64>
   {
      // computes weights to give each training sample
      let mut weights = matrix::make_covariance_matrix(&self.training_inputs, &inputs, &self.kernel, 0f64);
      self.covmat_cholesky.solve_mut(&mut weights);

      // computes prior for the given inputs
      let mut prior = self.prior.prior(&inputs);

      // weights.transpose() * &self.training_outputs + prior
      prior.gemm_tr(1f64, &weights, &self.training_outputs, 1f64);
      prior
   }

   /// predicts the covariance of the gaussian process at each row of the input
   ///
   /// NOTE:
   /// - combined with the mean, the function can be used to sample from the system
   /// TODO output struct with sample function (RNG->output) and mean/cov public members
   pub fn predict_covariance(&self, inputs: &DMatrix<f64>) -> DMatrix<f64>
   {
      // There is a better formula available if one can solve system directly using a triangular matrix
      // let kl = self.covmat_cholesky.l().solve(cov_train_inputs);
      // cov_inputs_inputs - (kl.transpose() * kl)

      // compute the weights
      let cov_train_inputs =
         matrix::make_covariance_matrix(&self.training_inputs, &inputs, &self.kernel, 0f64);
      let weights = self.covmat_cholesky.solve(&cov_train_inputs);

      // computes the intra points covariance
      let mut cov_inputs_inputs = matrix::make_covariance_matrix(&inputs, &inputs, &self.kernel, 0f64);

      // cov_inputs_inputs - cov_train_inputs.transpose() * weights
      cov_inputs_inputs.gemm_tr(-1f64, &cov_train_inputs, &weights, 1f64);
      cov_inputs_inputs
   }

   /// predicts the variance of the gaussian process at each row of the input
   ///
   /// NOTE:
   /// - unless the kernel was fitted on a one dimensional output, the magnitude of the variance is not linked to the magnitude of the outputs
   /// - this function is useful for bayesian optimization
   pub fn predict_variance(&self, inputs: &DMatrix<f64>) -> DVector<f64>
   {
      // There is a better formula available if one can solve system directly using a triangular matrix
      // let kl = self.covmat_cholesky.l().solve(cov_train_inputs);
      // cov_inputs_inputs - (kl.transpose() * kl).diagonal()
      // note that here the diagonal is just the sum of the squares of the values in the columns of kl

      // compute the weights
      let cov_train_inputs =
         matrix::make_covariance_matrix(&self.training_inputs, &inputs, &self.kernel, 0f64);
      let weights = self.covmat_cholesky.solve(&cov_train_inputs);

      // (cov_inputs_inputs - cov_train_inputs.transpose() * weights).diagonal()
      let mut variances = DVector::<f64>::zeros(inputs.nrows());
      for i in 0..inputs.nrows()
      {
         // Note that this might be done with a zipped iterator
         let input = inputs.row(i);
         let base_cov = self.kernel.kernel(input, input);
         let predicted_cov = cov_train_inputs.column(i).dot(&weights.column(i));
         variances[i] = base_cov - predicted_cov;
      }
      variances
   }
}
