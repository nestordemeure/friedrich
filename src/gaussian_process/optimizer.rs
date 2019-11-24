//! Parameter optimization
//!
//! The fit of the parameters is done by gradient descent (using the ADAM algorithm) on the gradient of the marginal log-likelihood
//! (which let us use all the data without bothering with cross-validation)
//!
//! The fit of the noise is done in log scale as its magnitude is more likely to be wrong and matters more than its precise value
//!
//! TODO :
//! - the paper [Fast methods for training Gaussian processes on large datasets](https://arxiv.org/pdf/1604.01250.pdf)
//! introduces a way to compute the analyticaly optimal scale for the kernel
//! this could let us get rid of the noise parameter
//! - the current implementation is memory hungry and could clearly be optimized
//! - is it better (for perf) to optimize noe parameter at a time or all at once ?

use crate::parameters::{kernel::Kernel, prior::Prior};
use super::GaussianProcess;
use crate::algebra::{make_cholesky_covariance_matrix, make_gradient_covariance_matrices};

impl<KernelType: Kernel, PriorType: Prior> GaussianProcess<KernelType, PriorType>
{
   /// Computes the gradient of the marginal likelihood for the current value of each parameter
   /// The produced vector contains the graident per kernel parameter followed by the gradient for the noise parameter
   ///
   /// NOTE: the gradient given for the noise is given in log scale (it is the gradient for ln(noise))
   fn gradient_marginal_likelihood(&self) -> Vec<f64>
   {
      // formula: 1/2 ( transpose(alpha) * dp * alpha - trace(K^-1 * dp) )
      // K = cov(train,train)
      // alpha = K^-1 * output
      // dp = gradient(K, parameter)

      // needed for the per parameter gradient computation
      let cov_inv = self.covmat_cholesky.inverse();
      let alpha = &cov_inv * self.training_outputs.as_vector();

      // loop on the gradient matrix for each parameter
      let mut results = vec![];
      for cov_gradient in make_gradient_covariance_matrices(&self.training_inputs.as_matrix(), &self.kernel)
      {
         // transpose(alpha) * cov_gradient * alpha
         let data_fit: f64 = cov_gradient.column_iter()
                                         .zip(alpha.iter())
                                         .map(|(col, alpha_col)| alpha.dot(&col) * alpha_col)
                                         .sum();

         // trace(cov_inv * cov_gradient)
         let complexity_penalty: f64 =
            cov_inv.row_iter().zip(cov_gradient.column_iter()).map(|(c, d)| c.tr_dot(&d)).sum();

         results.push((data_fit - complexity_penalty) / 2.);
      }

      // adds the noise parameter
      // gradient(K, noise) = 2*noise*Id
      let data_fit = alpha.dot(&alpha);
      let complexity_penalty = cov_inv.trace();
      let noise_gradient = self.noise * (data_fit - complexity_penalty) / 2.;
      results.push(self.noise * noise_gradient); // additioal noise parameter due to logarithmtic scaling of noise parameter

      results
   }

   /// Returns a vector containing all the parameters of the kernel plus the noise
   /// in the same order as the outputs of the `gradient_marginal_likelihood` function
   ///
   /// NOTE: noise is given in logarithmic scale `ln(noise)` as it make more sense to optimize it in log space
   fn get_parameters(&self) -> Vec<f64>
   {
      let mut parameters = self.kernel.get_parameters();
      parameters.push(self.noise.ln());
      parameters
   }

   /// Sets all the parameters of the kernel plus the noise
   /// by reading them from a slice where they are in the same order as the outputs of the `gradient_marginal_likelihood` function
   ///
   /// NOTE: noise is taken in logarithmic scale `ln(noise)` as it make more sense to optimize it in log space
   fn set_parameters(&mut self, parameters: &[f64], set_noise: bool)
   {
      self.kernel.set_parameters(&parameters[..parameters.len() - 1]);
      if set_noise
      {
         self.noise =
            parameters.last().expect("set_parameters: there should be at least one, noise, parameter!").exp();
      }
      // retrains the model on new parameters
      self.covmat_cholesky =
         make_cholesky_covariance_matrix(&self.training_inputs.as_matrix(), &self.kernel, self.noise);
   }

   /// Fit parameters using a gradient descent algorithm.
   ///
   /// Runs for a maximum of `max_iter` iterations (100 is a good default value).
   /// Stops prematurely if all the composants of the gradient go below `convergence_fraction` time the value of their respectiv parameter (0.01 is a good default value).
   /// Does not fit the noise parameter if `fit_noise` is set to false
   fn optimize_parameters(&mut self, fit_noise: bool, max_iter: usize, convergence_fraction: f64)
   {
      // use the ADAM gradient descent algorithm
      // see [optimizing-gradient-descent](https://ruder.io/optimizing-gradient-descent/)
      // for a good point on current gradient descent algorithms

      // constant parameters
      let beta1 = 0.9;
      let beta2 = 0.999;
      let epsilon = 1e-8;
      let learning_rate = 0.1;

      let mut parameters = self.get_parameters();
      let mut mean_grad = vec![0.; parameters.len()];
      let mut var_grad = vec![0.; parameters.len()];
      for i in 1..=max_iter
      {
         let gradients = self.gradient_marginal_likelihood();

         let mut continue_search = false;
         for p in 0..parameters.len()
         {
            mean_grad[p] = beta1 * mean_grad[p] + (1. - beta1) * gradients[p];
            var_grad[p] = beta2 * var_grad[p] + (1. - beta2) * gradients[p].powi(2);
            let bias_corrected_mean = mean_grad[p] / (1. - beta1.powi(i as i32));
            let bias_corrected_variance = var_grad[p] / (1. - beta2.powi(i as i32));
            let delta = learning_rate * bias_corrected_mean / (bias_corrected_variance.sqrt() + epsilon);
            continue_search |= delta.abs() > convergence_fraction;
            parameters[p] *= 1. + delta;
         }

         self.set_parameters(&parameters, fit_noise);
         if !continue_search
         {
            //println!("Fit done. iterations:{} likelihood:{} parameters:{:?}", i, self.likelihood(), parameters);
            break;
         };
      }
   }

   /// Fits the requested parameters and retrains the model.
   ///
   /// The fit of the noise and kernel parameters is done by gradient descent.
   /// It runs for a maximum of `max_iter` iterations and stops prematurely if all gradients are below `convergence_fraction` time their associated parameter.
   ///
   /// You cannot fit the noise parameter without also fitting the kernel.
   /// We recommend setting `fit_noise` to true when you fit the kernel unless you have a very good estimate of the noise in the output data.
   /// Note that if the `noise` parameter ends up unnaturaly large after the fit, it is a good sign that the kernel is unadapted to the data.
   ///
   /// Good base values for `max_iter` and `convergence_fraction` are 100 and 0.01
   pub fn fit_parameters(&mut self,
                         fit_prior: bool,
                         fit_kernel: bool,
                         fit_noise: bool,
                         max_iter: usize,
                         convergence_fraction: f64)
   {
      assert!(!fit_noise || fit_kernel,
              "fit_parameters: You cannot fit the noise without also fitting the kernel parameters.");

      if fit_prior
      {
         // gets the original data back in order to update the prior
         let training_outputs =
            self.training_outputs.as_vector() + self.prior.prior(&self.training_inputs.as_matrix());
         self.prior.fit(&self.training_inputs.as_matrix(), &training_outputs);
         let training_outputs = training_outputs - self.prior.prior(&self.training_inputs.as_matrix());
         self.training_outputs.assign(&training_outputs);
         // NOTE: adding and substracting each time we fit a prior might be numerically unwise

         if !fit_kernel
         {
            // retrains model from scratch
            self.covmat_cholesky =
               make_cholesky_covariance_matrix(&self.training_inputs.as_matrix(), &self.kernel, self.noise);
         }
      }

      if fit_kernel
      {
         // fit kernel and retrains model from scratch
         self.optimize_parameters(fit_noise, max_iter, convergence_fraction);
      }
   }
}
