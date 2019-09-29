//! Trained Gaussian process

use nalgebra::{DVector, DMatrix, Cholesky, Dynamic};
use crate::conversion::{AsMatrix, AsVector};
use crate::parameters::kernel::Kernel;
use crate::parameters::prior::Prior;
use crate::algebra;
use crate::algebra::{EMatrix, EVector};
use crate::multivariate_normal::MultivariateNormal;

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
   training_inputs: EMatrix,
   training_outputs: EVector,
   /// cholesky decomposition of the covariance matrix trained on the current datapoints
   covmat_cholesky: Cholesky<f64, Dynamic>
}

impl<KernelType: Kernel, PriorType: Prior> GaussianProcessTrained<KernelType, PriorType>
{
   pub fn new<InMatrix: AsMatrix, OutVector: AsVector>(prior: PriorType,
                                                       kernel: KernelType,
                                                       noise: f64,
                                                       training_inputs: InMatrix,
                                                       training_outputs: OutVector)
                                                       -> Self
   {
      // converts inputs into nalgebra format
      let training_inputs = training_inputs.as_matrix();
      let training_outputs = training_outputs.as_vector();
      // converts training data into extendable matrix
      let training_inputs = EMatrix::new(training_inputs);
      let training_outputs = EVector::new(training_outputs - prior.prior(&training_inputs.as_matrix()));
      // computes cholesky decomposition
      let covmat_cholesky =
         algebra::make_cholesky_covariance_matrix(&training_inputs.as_matrix(), &kernel, noise);
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
   pub fn add_samples<InMatrix: AsMatrix, OutVector: AsVector>(&mut self,
                                                               inputs: InMatrix,
                                                               outputs: OutVector)
   {
      // converts inputs into nalgebra format
      let inputs = inputs.as_matrix();
      let outputs = outputs.as_vector();
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

   /// fits the parameters and retrain the model from scratch
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

      self.covmat_cholesky = algebra::make_cholesky_covariance_matrix(&self.training_inputs.as_matrix(),
                                                                      &self.kernel,
                                                                      self.noise);
   }

   /// adds new samples to the model and fit the parameters
   /// faster than doing add_samples().fit_parameters()
   pub fn add_samples_fit<InMatrix: AsMatrix, OutVector: AsVector>(&mut self,
                                                                   inputs: InMatrix,
                                                                   outputs: OutVector,
                                                                   fit_prior: bool,
                                                                   fit_kernel: bool)
   {
      // converts inputs into nalgebra format
      let inputs = inputs.as_matrix();
      let outputs = outputs.as_vector();
      // grows the training matrix
      let outputs = outputs - self.prior.prior(&inputs);
      self.training_inputs.add_rows(&inputs);
      self.training_outputs.add_rows(&outputs);
      // refit the parameters and retrain the model from scratch
      self.fit_parameters(fit_prior, fit_kernel);
   }

   //----------------------------------------------------------------------------------------------
   // PREDICTION

   /// predicts the mean of the gaussian process at each row of the input
   pub fn predict_mean<'a, InMatrix>(&self, inputs: &'a InMatrix) -> DVector<f64>
      where &'a InMatrix: AsMatrix
   {
      // converts inputs into nalgebra format
      let inputs = inputs.as_matrix();

      // computes weights to give each training sample
      let mut weights =
         algebra::make_covariance_matrix(&self.training_inputs.as_matrix(), &inputs, &self.kernel);
      self.covmat_cholesky.solve_mut(&mut weights);

      // computes prior for the given inputs
      let mut prior = self.prior.prior(&inputs);

      // weights.transpose() * &self.training_outputs + prior
      prior.gemm_tr(1f64, &weights, &self.training_outputs.as_vector(), 1f64);
      prior
   }

   /// predicts the variance of the gaussian process at each row of the input
   ///
   /// NOTE:
   /// - this function is useful for bayesian optimization
   pub fn predict_variance<'a, InMatrix>(&self, inputs: &'a InMatrix) -> DVector<f64>
      where &'a InMatrix: AsMatrix
   {
      // There is a better formula available if one can solve system directly using a triangular matrix
      // let kl = self.covmat_cholesky.l().solve(cov_train_inputs);
      // cov_inputs_inputs - (kl.transpose() * kl).diagonal()
      // note that here the diagonal is just the sum of the squares of the values in the columns of kl

      // converts inputs into nalgebra format
      let inputs = inputs.as_matrix();

      // compute the weights
      let cov_train_inputs =
         algebra::make_covariance_matrix(&self.training_inputs.as_matrix(), &inputs, &self.kernel);
      let weights = self.covmat_cholesky.solve(&cov_train_inputs);

      // (cov_inputs_inputs - cov_train_inputs.transpose() * weights).diagonal()
      let mut variances = DVector::<f64>::zeros(inputs.nrows());
      for i in 0..inputs.nrows()
      {
         // Note that this might be done with a zipped iterator
         let input = inputs.row(i);
         let base_cov = self.kernel.kernel(&input, &input);
         let predicted_cov = cov_train_inputs.column(i).dot(&weights.column(i));
         variances[i] = base_cov - predicted_cov;
      }
      variances
   }

   /// predicts the std of the gaussian process at each row of the input
   ///
   /// NOTE:
   /// - this function is useful for bayesian optimization
   pub fn predict_standard_deviation<'a, InMatrix>(&self, inputs: &'a InMatrix) -> DVector<f64>
      where &'a InMatrix: AsMatrix
   {
      self.predict_variance(inputs).apply_into(|x| x.sqrt())
   }

   /// predicts the covariance of the gaussian process at each row of the input
   pub fn predict_covariance<'a, InMatrix>(&self, inputs: &'a InMatrix) -> DMatrix<f64>
      where &'a InMatrix: AsMatrix
   {
      // There is a better formula available if one can solve system directly using a triangular matrix
      // let kl = self.covmat_cholesky.l().solve(cov_train_inputs);
      // cov_inputs_inputs - (kl.transpose() * kl)

      // converts inputs into nalgebra format
      let inputs = inputs.as_matrix();

      // compute the weights
      let cov_train_inputs =
         algebra::make_covariance_matrix(&self.training_inputs.as_matrix(), &inputs, &self.kernel);
      let weights = self.covmat_cholesky.solve(&cov_train_inputs);

      // computes the intra points covariance
      let mut cov_inputs_inputs = algebra::make_covariance_matrix(&inputs, &inputs, &self.kernel);

      // cov_inputs_inputs - cov_train_inputs.transpose() * weights
      cov_inputs_inputs.gemm_tr(-1f64, &cov_train_inputs, &weights, 1f64);
      cov_inputs_inputs
   }

   /// produces a structure that can be used to sample the gaussian process at the given points
   pub fn sample_at<'a, InMatrix>(&self, inputs: &'a InMatrix) -> MultivariateNormal
      where &'a InMatrix: AsMatrix
   {
      // TODO we can factor some operations and improve performance by inlining and fusing the function needed
      let mean = self.predict_mean(inputs);
      let cov_inputs = self.predict_covariance(inputs);
      MultivariateNormal::new(mean, cov_inputs)
   }
}
