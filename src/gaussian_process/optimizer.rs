//! Parameter optimization
//!
//! The fit of the parameters is done by gradient descent (using the ADAM algorithm) on the gradient of the marginal log-likelihood
//! (which let us use all the data without bothering with cross-validation)
//!
//! If the kernel can be rescaled, we use ideas from [Fast methods for training Gaussian processes on large datasets](https://arxiv.org/pdf/1604.01250.pdf)
//! to rescale the kernel at each step with the optimal magnitude which has the effect of fitting the noise without computing its gradient.
//!
//! Otherwise we fit the noise in log-scale as its magnitude matters more than its precise value.
//!
//! TODO :
//! - the current implementation of `gradient_marginal_likelihood` is memory hungry and could clearly be optimized
//! - is it better (for perf) to optimize one parameter at a time or all at once ?

use crate::parameters::{kernel::Kernel, prior::Prior};
use super::GaussianProcess;
use crate::algebra::{make_cholesky_cov_matrix, make_gradient_covariance_matrices};

impl<KernelType: Kernel, PriorType: Prior> GaussianProcess<KernelType, PriorType>
{
   //-------------------------------------------------------------------------------------------------
   // NON-SCALABLE KERNEL

   /// Computes the gradient of the marginal likelihood for the current value of each parameter
   /// The produced vector contains the gradient per kernel parameter followed by the gradient for the noise parameter
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
      let noise_gradient = self.noise * (data_fit - complexity_penalty);
      results.push(noise_gradient);

      results
   }

   /// Fit parameters using a gradient descent algorithm.
   ///
   /// Runs for a maximum of `max_iter` iterations (100 is a good default value).
   /// Stops prematurely if all the composants of the gradient go below `convergence_fraction` time the value of their respectiv parameter (0.05 is a good default value).
   ///
   /// The `noise` parameter is fitted in log-scale as its magnitude matters more than its precise value
   fn optimize_parameters(&mut self, max_iter: usize, convergence_fraction: f64)
   {
      // use the ADAM gradient descent algorithm
      // see [optimizing-gradient-descent](https://ruder.io/optimizing-gradient-descent/)
      // for a good point on current gradient descent algorithms

      // constant parameters
      let beta1 = 0.9;
      let beta2 = 0.999;
      let epsilon = 1e-8;
      let learning_rate = 0.1;

      let mut parameters = self.kernel.get_parameters();
      parameters.push(self.noise.ln()); // adds noise in log-space

      let mut mean_grad = vec![0.; parameters.len()];
      let mut var_grad = vec![0.; parameters.len()];
      for i in 1..=max_iter
      {
         let mut gradients = self.gradient_marginal_likelihood();
         gradients.last_mut().map(|noise_grad| *noise_grad *= self.noise); // corrects gradient of noise for log-space

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

         // sets parameters
         self.kernel.set_parameters(&parameters);
         parameters.last().map(|noise| self.noise = noise.exp()); // gets out of log-space before setting noise

         // fits model
         self.covmat_cholesky =
            make_cholesky_cov_matrix(&self.training_inputs.as_matrix(), &self.kernel, self.noise);

         if !continue_search
         {
            //println!("Iterations:{}", i);
            break;
         };
      }

      /*println!("Fit done. likelihood:{} parameters:{:?} noise:{:e}",
      self.likelihood(),
      parameters,
      self.noise);*/
   }

   //-------------------------------------------------------------------------------------------------
   // SCALABLE KERNEL

   /// Returns a couple containing the optimal scale for the kernel+noise (which is used to optimize the noise)
   /// plus a vector containing the gradient per kernel parameter (but NOT the gradient for the noise parameter)
   ///
   /// see [Fast methods for training Gaussian processes on large datasets](https://arxiv.org/pdf/1604.01250.pdf)
   /// for the formula used to compute the scale and the modification to the gradient
   fn scaled_gradient_marginal_likelihood(&self) -> (f64, Vec<f64>)
   {
      // formula:
      // gradient = 1/2 ( transpose(alpha) * dp * alpha / scale - trace(K^-1 * dp) )
      // scale = transpose(output) * K^-1 * output / n
      // K = cov(train,train)
      // alpha = K^-1 * output
      // dp = gradient(K, parameter)

      // needed for the per parameter gradient computation
      let cov_inv = self.covmat_cholesky.inverse();
      let training_output = self.training_outputs.as_vector();
      let alpha = &cov_inv * training_output;

      // scaling for the kernel
      let scale = training_output.dot(&alpha) / (training_output.nrows() as f64);

      // loop on the gradient matrix for each parameter
      let mut results = vec![];
      for cov_gradient in make_gradient_covariance_matrices(&self.training_inputs.as_matrix(), &self.kernel)
      {
         // transpose(alpha) * cov_gradient * alpha / scale
         // NOTE: this quantity is divided by the scale wich is not the case for the unscaled gradient
         let data_fit = cov_gradient.column_iter()
                                    .zip(alpha.iter())
                                    .map(|(col, alpha_col)| alpha.dot(&col) * alpha_col)
                                    .sum::<f64>()
                        / scale;

         // trace(cov_inv * cov_gradient)
         let complexity_penalty: f64 =
            cov_inv.row_iter().zip(cov_gradient.column_iter()).map(|(c, d)| c.tr_dot(&d)).sum();

         results.push((data_fit - complexity_penalty) / 2.);
      }

      // adds the noise parameter
      // gradient(K, noise) = 2*noise*Id
      /*let data_fit = alpha.dot(&alpha) / scale;
      let complexity_penalty = cov_inv.trace();
      let noise_gradient = self.noise * (data_fit - complexity_penalty);
      results.push(noise_gradient);*/

      (scale, results)
   }

   /// Fit parameters using a gradient descent algorithm.
   /// Additionnaly, at eac step, the kernel and noise are rescaled using the optimal magnitude.
   ///
   /// Runs for a maximum of `max_iter` iterations (100 is a good default value).
   /// Stops prematurely if all the composants of the gradient go below `convergence_fraction` time the value of their respectiv parameter (0.05 is a good default value).
   fn scaled_optimize_parameters(&mut self, max_iter: usize, convergence_fraction: f64)
   {
      // use the ADAM gradient descent algorithm
      // see [optimizing-gradient-descent](https://ruder.io/optimizing-gradient-descent/)
      // for a good point on current gradient descent algorithms

      // constant parameters
      let beta1 = 0.9;
      let beta2 = 0.999;
      let epsilon = 1e-8;
      let learning_rate = 0.1;

      let mut parameters = self.kernel.get_parameters();
      let mut mean_grad = vec![0.; parameters.len()];
      let mut var_grad = vec![0.; parameters.len()];
      for i in 1..=max_iter
      {
         let (scale, gradients) = self.scaled_gradient_marginal_likelihood();

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

         // set parameters
         self.kernel.set_parameters(&parameters);
         self.kernel.rescale(scale);
         self.noise *= scale;
         parameters = self.kernel.get_parameters(); // get parameters back as they have been rescaled

         // fits model
         self.covmat_cholesky =
            make_cholesky_cov_matrix(&self.training_inputs.as_matrix(), &self.kernel, self.noise);

         if !continue_search
         {
            //println!("Iterations:{}", i);
            break;
         };
      }

      /*println!("Scaled fit done. likelihood:{} parameters:{:?} noise:{:e}",
      self.likelihood(),
      parameters,
      self.noise);*/
   }

   //-------------------------------------------------------------------------------------------------
   // NON-SCALABLE KERNEL

   /// Fits the requested parameters and retrains the model.
   ///
   /// The fit of the noise and kernel parameters is done by gradient descent.
   /// It runs for a maximum of `max_iter` iterations and stops prematurely if all gradients are below `convergence_fraction` time their associated parameter.
   ///
   /// Good base values for `max_iter` and `convergence_fraction` are 100 and 0.05
   ///
   /// Note that, if the `noise` parameter ends up unnaturaly large after the fit, it is a good sign that the kernel is unadapted to the data.
   pub fn fit_parameters(&mut self,
                         fit_prior: bool,
                         fit_kernel: bool,
                         max_iter: usize,
                         convergence_fraction: f64)
   {
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
               make_cholesky_cov_matrix(&self.training_inputs.as_matrix(), &self.kernel, self.noise);
         }
      }

      // fit kernel and retrains model from scratch
      if fit_kernel
      {
         if KernelType::IS_SCALABLE
         {
            self.scaled_optimize_parameters(max_iter, convergence_fraction);
         }
         else
         {
            self.optimize_parameters(max_iter, convergence_fraction);
         }
      }
   }
}
