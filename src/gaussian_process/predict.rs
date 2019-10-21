//! Methods to make predictions with a gaussian process.

use nalgebra::DMatrix;
use crate::parameters::kernel::Kernel;
use crate::parameters::prior::Prior;
use crate::algebra;
use super::GaussianProcess;

impl<KernelType: Kernel, PriorType: Prior> GaussianProcess<KernelType, PriorType>
{
   /// predicts the mean of the gaussian process for an input
   pub fn predict(&self, input: &[f64]) -> f64
   {
      let input = DMatrix::from_row_slice(1, input.len(), input);
      let result = self.gp.predict(&input);
      result[0]
   }

   /// predicts the mean of the gaussian process at each row of the input
   pub fn predict_several(&self, inputs: &[Vec<f64>]) -> Vec<f64>
   {
      let inputs = algebra::make_matrix_from_row_slices(inputs);
      let result = self.gp.predict(&inputs);
      result.iter().cloned().collect()
   }

   /// predicts the variance of the gaussian process for an input
   pub fn predict_variance(&self, input: &[f64]) -> f64
   {
      let input = DMatrix::from_row_slice(1, input.len(), input);
      let result = self.gp.predict_variance(&input);
      result[0]
   }

   /// predicts the variance of the gaussian process at each row of the input
   pub fn predict_variance_several(&self, inputs: &[Vec<f64>]) -> Vec<f64>
   {
      let inputs = algebra::make_matrix_from_row_slices(inputs);
      let result = self.gp.predict_variance(&inputs);
      result.iter().cloned().collect()
   }

   /// predicts the covariance of the gaussian process at each row of the input
   pub fn predict_covariance_several(&self, inputs: &[Vec<f64>]) -> DMatrix<f64>
   {
      let inputs = algebra::make_matrix_from_row_slices(inputs);
      self.gp.predict_covariance(&inputs)
   }

   /// produces a structure that can be used to sample the gaussian process at the given points
   pub fn sample_at_several(&self, inputs: &[Vec<f64>]) -> algebra::MultivariateNormal
   {
      let inputs = algebra::make_matrix_from_row_slices(inputs);
      self.gp.sample_at(&inputs)
   }
}
