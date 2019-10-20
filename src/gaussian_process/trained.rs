//! Trained Gaussian process

use std::marker::PhantomData;
use nalgebra::{DVector, DMatrix, Cholesky, Dynamic};
use crate::conversion::{AsMatrix, AsVector};
use crate::parameters::kernel::Kernel;
use crate::parameters::prior::Prior;
use crate::algebra;
use crate::algebra::{EMatrix, EVector, MultivariateNormal};

/// gaussian process
pub struct GaussianProcessTrained<KernelType: Kernel, PriorType: Prior, OutVector: AsVector>
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
   /// types of the inputs and outputs
   output_type: PhantomData<OutVector>,
   /// cholesky decomposition of the covariance matrix trained on the current datapoints
   covmat_cholesky: Cholesky<f64, Dynamic>
}

impl<KernelType: Kernel, PriorType: Prior, OutVector: AsVector>
   GaussianProcessTrained<KernelType, PriorType, OutVector>
{
   pub fn new<InMatrix: AsMatrix>(prior: PriorType,
                                  kernel: KernelType,
                                  noise: f64,
                                  training_inputs: InMatrix,
                                  training_outputs: OutVector)
                                  -> Self
   {
      // converts inputs into nalgebra format
      let training_inputs = training_inputs.into_matrix();
      let training_outputs = training_outputs.into_vector();
      assert_eq!(training_inputs.nrows(), training_outputs.nrows());
      // converts training data into extendable matrix
      let training_inputs = EMatrix::new(training_inputs);
      let training_outputs = EVector::new(training_outputs - prior.prior(&training_inputs.as_matrix()));
      // computes cholesky decomposition
      let covmat_cholesky =
         algebra::make_cholesky_covariance_matrix(&training_inputs.as_matrix(), &kernel, noise);
      GaussianProcessTrained { prior,
                               kernel,
                               noise,
                               training_inputs,
                               training_outputs,
                               covmat_cholesky,
                               output_type: PhantomData }
   }

   //----------------------------------------------------------------------------------------------
   // TRAINING

   /// adds new samples to the model
   /// update the model (which is faster than a training from scratch)
   /// does not refit the parameters
   pub fn add_samples_several<InMatrix: AsMatrix, OutVector2: AsVector>(&mut self,
                                                                        inputs: &InMatrix,
                                                                        outputs: &OutVector2)
   {
      // converts inputs into nalgebra format
      let inputs = inputs.as_matrix();
      let outputs = outputs.as_vector();
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
   pub fn add_sample<InVector: AsVector>(&mut self, input: &InVector, output: f64)
   {
      let input = input.as_vector();
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
   pub fn add_samples_fit_several<InMatrix: AsMatrix, OutVector2: AsVector>(&mut self,
                                                                            inputs: &InMatrix,
                                                                            outputs: &OutVector2,
                                                                            fit_prior: bool,
                                                                            fit_kernel: bool)
   {
      // converts inputs into nalgebra format
      let inputs = inputs.as_matrix();
      let outputs = outputs.as_vector();
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
   pub fn add_sample_fit<InVector: AsVector>(&mut self,
                                             input: &InVector,
                                             output: f64,
                                             fit_prior: bool,
                                             fit_kernel: bool)
   {
      let input = input.as_vector();
      let input = DMatrix::from_row_slice(1, input.nrows(), input.as_slice());
      let output = DVector::from_element(1, output);
      self.add_samples_fit_several(&input, &output, fit_prior, fit_kernel)
   }

   //----------------------------------------------------------------------------------------------
   // PREDICTION

   /// predicts the mean of the gaussian process for an input
   pub fn predict<InVector: AsVector>(&self, input: &InVector) -> f64
   {
      let input = input.as_vector();
      let input = DMatrix::from_row_slice(1, input.nrows(), input.as_slice());
      let result = self.predict_several(&input);
      result.as_vector()[0]
   }

   /// predicts the mean of the gaussian process at each row of the input
   pub fn predict_several<InMatrix: AsMatrix>(&self, inputs: &InMatrix) -> OutVector
   {
      // converts inputs into nalgebra format
      let inputs = inputs.as_matrix();
      assert_eq!(inputs.ncols(), self.training_inputs.as_matrix().ncols());

      // computes weights to give each training sample
      let mut weights =
         algebra::make_covariance_matrix(&self.training_inputs.as_matrix(), &inputs, &self.kernel);
      self.covmat_cholesky.solve_mut(&mut weights);

      // computes prior for the given inputs
      let mut prior = self.prior.prior(&inputs);

      // weights.transpose() * &self.training_outputs + prior
      prior.gemm_tr(1f64, &weights, &self.training_outputs.as_vector(), 1f64);

      // converts to expected output type
      OutVector::from_vector(prior)
   }

   /// predicts the variance of the gaussian process for an input
   pub fn predict_variance<InVector: AsVector>(&self, input: &InVector) -> f64
   {
      let input = input.as_vector();
      let input = DMatrix::from_row_slice(1, input.nrows(), input.as_slice());
      let result = self.predict_variance_several(&input);
      result.as_vector()[0]
   }

   /// predicts the variance of the gaussian process at each row of the input
   pub fn predict_variance_several<InMatrix: AsMatrix>(&self, inputs: &InMatrix) -> OutVector
   {
      // There is a better formula available if one can solve system directly using a triangular matrix
      // let kl = self.covmat_cholesky.l().solve(cov_train_inputs);
      // cov_inputs_inputs - (kl.transpose() * kl).diagonal()
      // note that here the diagonal is just the sum of the squares of the values in the columns of kl

      // converts inputs into nalgebra format
      let inputs = inputs.as_matrix();
      assert_eq!(inputs.ncols(), self.training_inputs.as_matrix().ncols());

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

      OutVector::from_vector(variances)
   }

   /// predicts the covariance of the gaussian process at each row of the input
   pub fn predict_covariance_several<InMatrix: AsMatrix>(&self, inputs: &InMatrix) -> DMatrix<f64>
   {
      // There is a better formula available if one can solve system directly using a triangular matrix
      // let kl = self.covmat_cholesky.l().solve(cov_train_inputs);
      // cov_inputs_inputs - (kl.transpose() * kl)

      // converts inputs into nalgebra format
      let inputs = inputs.as_matrix();
      assert_eq!(inputs.ncols(), self.training_inputs.as_matrix().ncols());

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
   pub fn sample_at_several<InMatrix: AsMatrix>(&self, inputs: &InMatrix) -> MultivariateNormal<OutVector>
   {
      // converts inputs into nalgebra format
      let inputs = inputs.as_matrix();
      assert_eq!(inputs.ncols(), self.training_inputs.as_matrix().ncols());

      // compute the weights
      let cov_train_inputs =
         algebra::make_covariance_matrix(&self.training_inputs.as_matrix(), &inputs, &self.kernel);
      let weights = self.covmat_cholesky.solve(&cov_train_inputs);

      // computes covariance
      let mut cov_inputs_inputs = algebra::make_covariance_matrix(&inputs, &inputs, &self.kernel);
      cov_inputs_inputs.gemm_tr(-1f64, &cov_train_inputs, &weights, 1f64);
      let cov = cov_inputs_inputs;

      // computes the mean
      let mut prior = self.prior.prior(&inputs);
      prior.gemm_tr(1f64, &weights, &self.training_outputs.as_vector(), 1f64);
      let mean = prior;

      MultivariateNormal::new(mean, cov)
   }
}
