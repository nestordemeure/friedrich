//! Methods to make predictions with a gaussian process.

use nalgebra::{DVector, DMatrix, RowDVector};
use crate::parameters::kernel::Kernel;
use crate::parameters::prior::Prior;
use crate::algebra;
use crate::algebra::MultivariateNormal;
use super::GaussianProcess_nalgebra;

impl<KernelType: Kernel, PriorType: Prior> GaussianProcess_nalgebra<KernelType, PriorType>
{
   /// predicts the mean of the gaussian process for an input
   pub fn predict(&self, input: &RowDVector<f64>) -> f64
   {
      let input = DMatrix::from_row_slice(1, input.ncols(), input.as_slice());
      let result = self.predict_several(&input);
      result[0]
   }

   /// predicts the mean of the gaussian process at each row of the input
   pub fn predict_several(&self, inputs: &DMatrix<f64>) -> DVector<f64>
   {
      assert_eq!(inputs.ncols(), self.training_inputs.as_matrix().ncols());

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

   /// predicts the variance of the gaussian process for an input
   pub fn predict_variance(&self, input: &RowDVector<f64>) -> f64
   {
      let input = DMatrix::from_row_slice(1, input.ncols(), input.as_slice());
      let result = self.predict_variance_several(&input);
      result[0]
   }

   /// predicts the variance of the gaussian process at each row of the input
   pub fn predict_variance_several(&self, inputs: &DMatrix<f64>) -> DVector<f64>
   {
      // There is a better formula available if one can solve system directly using a triangular matrix
      // let kl = self.covmat_cholesky.l().solve(cov_train_inputs);
      // cov_inputs_inputs - (kl.transpose() * kl).diagonal()
      // note that here the diagonal is just the sum of the squares of the values in the columns of kl

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

      variances
   }

   /// predicts the covariance of the gaussian process at each row of the input
   pub fn predict_covariance_several(&self, inputs: &DMatrix<f64>) -> DMatrix<f64>
   {
      // There is a better formula available if one can solve system directly using a triangular matrix
      // let kl = self.covmat_cholesky.l().solve(cov_train_inputs);
      // cov_inputs_inputs - (kl.transpose() * kl)

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
   pub fn sample_at_several(&self, inputs: &DMatrix<f64>) -> MultivariateNormal<DVector<f64>>
   {
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
