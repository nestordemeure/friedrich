//! Gaussian process
//!
//! ```rust
//! # use friedrich::gaussian_process::GaussianProcess;
//! # fn main() {
//! // trains a gaussian process on a dataset of one dimension vectors
//! let training_inputs = vec![vec![0.8], vec![1.2], vec![3.8], vec![4.2]];
//! let training_outputs = vec![3.0, 4.0, -2.0, -2.0];
//! let mut gp = GaussianProcess::default(training_inputs, training_outputs);
//!
//! // predicts the mean and variance of a single point
//! let input = vec![1.];
//! let mean = gp.predict(&input);
//! let var = gp.predict_variance(&input);
//! println!("prediction: {} ± {}", mean, var.sqrt());
//!
//! // makes several prediction
//! let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
//! let outputs = gp.predict(&inputs);
//! println!("predictions: {:?}", outputs);
//!
//! // updates the model
//! let additional_inputs = vec![vec![0.], vec![1.], vec![2.], vec![5.]];
//! let additional_outputs = vec![2.0, 3.0, -1.0, -2.0];
//! let fit_prior = true;
//! let fit_kernel = true;
//! gp.add_samples_fit(&additional_inputs, &additional_outputs, fit_prior, fit_kernel);
//!
//! // samples from the distribution
//! let new_inputs = vec![vec![1.0], vec![2.0]];
//! let sampler = gp.sample_at(&new_inputs);
//! let mut rng = rand::thread_rng();
//! for i in 1..=5
//! {
//!   println!("sample {} : {:?}", i, sampler.sample(&mut rng));
//! }
//! # }
//! ```

use crate::parameters::{kernel, kernel::Kernel, prior, prior::Prior};
use nalgebra::{Cholesky, Dynamic, DMatrix, DVector};
use crate::algebra::{EMatrix, EVector, make_cholesky_covariance_matrix, make_covariance_matrix};
use crate::conversion::Input;

mod multivariate_normal;
pub use multivariate_normal::MultivariateNormal;

mod builder;
pub use builder::GaussianProcessBuilder;

/// A Gaussian process that can be used to make predictions based on its training data
pub struct GaussianProcess<KernelType: Kernel, PriorType: Prior>
{
   /// value to which the process will regress in the absence of informations
   pub prior: PriorType,
   /// kernel used to fit the process on the data
   pub kernel: KernelType,
   /// amplitude of the noise of the data
   pub noise: f64,
   /// data used for fit
   training_inputs: EMatrix,
   training_outputs: EVector,
   /// cholesky decomposition of the covariance matrix trained on the current datapoints
   covmat_cholesky: Cholesky<f64, Dynamic>
}

impl GaussianProcess<kernel::Gaussian, prior::ConstantPrior>
{
   /// Returns a gaussian process with a Gaussian kernel and a constant prior, both fitted to the data.
   ///
   /// ```rust
   /// # use friedrich::gaussian_process::GaussianProcess;
   /// # fn main() {
   /// // training data
   /// let training_inputs = vec![vec![0.8], vec![1.2], vec![3.8], vec![4.2]];
   /// let training_outputs = vec![3.0, 4.0, -2.0, -2.0];
   ///
   /// // defining and training a model
   /// let gp = GaussianProcess::default(training_inputs, training_outputs);
   /// # }
   /// ```
   pub fn default<T: Input>(training_inputs: T, training_outputs: T::InVector) -> Self
   {
      GaussianProcessBuilder::<kernel::Gaussian, prior::ConstantPrior>::new(training_inputs, training_outputs)
      .fit_kernel()
      .fit_prior()
      .train()
   }

   /// Returns a builder to define specific parameters of the gaussian process.
   ///
   /// ```rust
   /// # use friedrich::gaussian_process::GaussianProcess;
   /// # use friedrich::prior::*;
   /// # use friedrich::kernel::*;
   /// # fn main() {
   /// # // training data
   /// # let training_inputs = vec![vec![0.8], vec![1.2], vec![3.8], vec![4.2]];
   /// # let training_outputs = vec![3.0, 4.0, -2.0, -2.0];
   /// // model parameters
   /// let input_dimension = 1;
   /// let output_noise = 0.1;
   /// let exponential_kernel = Exponential::default();
   /// let linear_prior = LinearPrior::default(input_dimension);
   ///
   /// // defining and training a model
   /// let gp = GaussianProcess::builder(training_inputs, training_outputs).set_noise(output_noise)
   ///                                                                     .set_kernel(exponential_kernel)
   ///                                                                     .fit_kernel()
   ///                                                                     .set_prior(linear_prior)
   ///                                                                     .fit_prior()
   ///                                                                     .train();
   /// # }
   /// ```
   pub fn builder<T: Input>(training_inputs: T,
                            training_outputs: T::InVector)
                            -> GaussianProcessBuilder<kernel::Gaussian, prior::ConstantPrior>
   {
      GaussianProcessBuilder::<kernel::Gaussian, prior::ConstantPrior>::new(training_inputs, training_outputs)
   }
}

impl<KernelType: Kernel, PriorType: Prior> GaussianProcess<KernelType, PriorType>
{
   /// Raw method to create a new gaussian process with the given parameters / data.
   /// We recommend that you use either the default parameters or the builder to simplify the definition process.
   pub fn new<T: Input>(prior: PriorType,
                        kernel: KernelType,
                        noise: f64,
                        training_inputs: T,
                        training_outputs: T::InVector)
                        -> Self
   {
      let training_inputs = T::into_dmatrix(training_inputs);
      let training_outputs = T::into_dvector(training_outputs);
      assert_eq!(training_inputs.nrows(), training_outputs.nrows());
      // converts training data into extendable matrix
      let training_inputs = EMatrix::new(training_inputs);
      let training_outputs = EVector::new(training_outputs - prior.prior(&training_inputs.as_matrix()));
      // computes cholesky decomposition
      let covmat_cholesky = make_cholesky_covariance_matrix(&training_inputs.as_matrix(), &kernel, noise);
      GaussianProcess { prior, kernel, noise, training_inputs, training_outputs, covmat_cholesky }
   }

   //----------------------------------------------------------------------------------------------
   // FIT

   /// Adds new samples to the model.
   ///
   /// Updates the model (which is faster than a retraining from scratch)
   /// but does not refit the parameters.
   pub fn add_samples<T: Input>(&mut self, inputs: &T, outputs: &T::InVector)
   {
      let inputs = T::to_dmatrix(inputs);
      let outputs = T::to_dvector(outputs);
      assert_eq!(inputs.nrows(), outputs.nrows());
      assert_eq!(inputs.ncols(), self.training_inputs.as_matrix().ncols());
      // grows the training matrix
      let outputs = outputs - self.prior.prior(&inputs);
      self.training_inputs.add_rows(&inputs);
      self.training_outputs.add_rows(&outputs);
      // recompute cholesky matrix
      self.covmat_cholesky =
         make_cholesky_covariance_matrix(&self.training_inputs.as_matrix(), &self.kernel, self.noise);
      // TODO update cholesky matrix instead of recomputing it from scratch
   }

   /// Fits the parameters, if requested, and retrain the model from scratch.
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
         self.covmat_cholesky =
            make_cholesky_covariance_matrix(&self.training_inputs.as_matrix(), &self.kernel, self.noise);
      }
   }

   /// Adds new samples to the model and fit the parameters.
   ///
   /// Faster than doing `add_samples().fit_parameters()`.
   pub fn add_samples_fit<T: Input>(&mut self,
                                    inputs: &T,
                                    outputs: &T::InVector,
                                    fit_prior: bool,
                                    fit_kernel: bool)
   {
      let inputs = T::to_dmatrix(inputs);
      let outputs = T::to_dvector(outputs);
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
         self.covmat_cholesky =
            make_cholesky_covariance_matrix(&self.training_inputs.as_matrix(), &self.kernel, self.noise);
      }
   }

   //----------------------------------------------------------------------------------------------
   // PREDICT

   /// Predicts the mean of the gaussian process for each row of the input.
   pub fn predict<T: Input>(&self, inputs: &T) -> T::OutVector
   {
      let inputs = T::to_dmatrix(inputs);
      assert_eq!(inputs.ncols(), self.training_inputs.as_matrix().ncols());

      // computes weights to give each training sample
      let mut weights = make_covariance_matrix(&self.training_inputs.as_matrix(), &inputs, &self.kernel);
      self.covmat_cholesky.solve_mut(&mut weights);

      // computes prior for the given inputs
      let mut prior = self.prior.prior(&inputs);

      // weights.transpose() * &self.training_outputs + prior
      prior.gemm_tr(1f64, &weights, &self.training_outputs.as_vector(), 1f64);

      T::from_dvector(&prior)
   }

   /// Predicts the variance of the gaussian process for each row of the input.
   pub fn predict_variance<T: Input>(&self, inputs: &T) -> T::OutVector
   {
      let inputs = T::to_dmatrix(inputs);
      assert_eq!(inputs.ncols(), self.training_inputs.as_matrix().ncols());

      // compute the covariances
      let cov_train_inputs = make_covariance_matrix(&self.training_inputs.as_matrix(), &inputs, &self.kernel);

      // solve linear system
      let kl = self.covmat_cholesky
                   .l()
                   .solve_lower_triangular(&cov_train_inputs)
                   .expect("predict_covariance : solve failed");

      // (cov_inputs_inputs - (kl.transpose() * kl)).diagonal()
      let variances = inputs.row_iter()
                            .map(|row| self.kernel.kernel(&row, &row)) // variance of input points with themselves
                            .zip(kl.column_iter().map(|col| col.norm_squared())) // diag(kl^T * kl)
                            .map(|(base_cov, predicted_cov)| base_cov - predicted_cov);
      let variances = DVector::<f64>::from_iterator(inputs.nrows(), variances);

      T::from_dvector(&variances)
   }

   /// Returns the covariancematrix for the rows of the input.
   pub fn predict_covariance<T: Input>(&self, inputs: &T) -> DMatrix<f64>
   {
      let inputs = T::to_dmatrix(inputs);
      assert_eq!(inputs.ncols(), self.training_inputs.as_matrix().ncols());

      // compute the covariances
      let cov_train_inputs = make_covariance_matrix(&self.training_inputs.as_matrix(), &inputs, &self.kernel);
      let mut cov_inputs_inputs = make_covariance_matrix(&inputs, &inputs, &self.kernel);

      // solve linear system
      let kl = self.covmat_cholesky
                   .l()
                   .solve_lower_triangular(&cov_train_inputs)
                   .expect("predict_covariance : solve failed");

      // cov_inputs_inputs - (kl.transpose() * kl)
      cov_inputs_inputs.gemm_tr(-1f64, &kl, &kl, 1f64);
      cov_inputs_inputs
   }

   /// Produces a multivariate gaussian that can be used to sample at the input points.
   ///
   /// The sampling is requires a random number generator compatible with the [rand](https://crates.io/crates/rand) crate :
   ///
   /// ```rust
   /// # use friedrich::gaussian_process::GaussianProcess;
   /// # fn main() {
   /// # let training_inputs = vec![vec![0.8], vec![1.2], vec![3.8], vec![4.2]];
   /// # let training_outputs = vec![3.0, 4.0, -2.0, -2.0];
   /// # let gp = GaussianProcess::default(training_inputs, training_outputs);
   /// // computes the distribution at some new coordinates
   /// let new_inputs = vec![vec![1.], vec![2.]];
   /// let sampler = gp.sample_at(&new_inputs);
   ///
   /// // samples from the distribution
   /// let mut rng = rand::thread_rng();
   /// println!("samples a vector : {:?}", sampler.sample(&mut rng));
   /// # }
   /// ```
   pub fn sample_at<T: Input>(&self, inputs: &T) -> MultivariateNormal<T>
   {
      let inputs = T::to_dmatrix(inputs);
      assert_eq!(inputs.ncols(), self.training_inputs.as_matrix().ncols());

      // compute the weights
      let cov_train_inputs = make_covariance_matrix(&self.training_inputs.as_matrix(), &inputs, &self.kernel);
      let weights = self.covmat_cholesky.solve(&cov_train_inputs);

      // computes covariance
      let mut cov_inputs_inputs = make_covariance_matrix(&inputs, &inputs, &self.kernel);
      cov_inputs_inputs.gemm_tr(-1f64, &cov_train_inputs, &weights, 1f64);
      let cov = cov_inputs_inputs;

      // computes the mean
      let mut prior = self.prior.prior(&inputs);
      prior.gemm_tr(1f64, &weights, &self.training_outputs.as_vector(), 1f64);
      let mean = prior;

      MultivariateNormal::new(mean, cov)
   }
}
