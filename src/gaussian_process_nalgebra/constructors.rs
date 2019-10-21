//! Methods to build a Gaussian process.

use super::NAlgebraGaussianProcess;
use crate::parameters::{kernel::Kernel, prior::Prior};
use crate::algebra::{EMatrix, EVector};
use crate::algebra;
use nalgebra::{DMatrix, DVector};

impl<KernelType: Kernel, PriorType: Prior> NAlgebraGaussianProcess<KernelType, PriorType>
{
   pub fn new(prior: PriorType,
              kernel: KernelType,
              noise: f64,
              training_inputs: DMatrix<f64>,
              training_outputs: DVector<f64>)
              -> Self
   {
      assert_eq!(training_inputs.nrows(), training_outputs.nrows());
      // converts training data into extendable matrix
      let training_inputs = EMatrix::new(training_inputs);
      let training_outputs = EVector::new(training_outputs - prior.prior(&training_inputs.as_matrix()));
      // computes cholesky decomposition
      let covmat_cholesky =
         algebra::make_cholesky_covariance_matrix(&training_inputs.as_matrix(), &kernel, noise);
      NAlgebraGaussianProcess { prior, kernel, noise, training_inputs, training_outputs, covmat_cholesky }
   }
}
