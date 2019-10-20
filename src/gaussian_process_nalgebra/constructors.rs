//! Methods to build a Gaussian process.

use super::GaussianProcess_nalgebra;
use super::builder::GaussianProcessBuilder_nalgebra;
use crate::parameters::{kernel::Kernel, prior::Prior};
use crate::parameters::*;
use crate::algebra::{EMatrix, EVector};
use crate::algebra;
use nalgebra::{DMatrix, DVector};

impl<KernelType: Kernel, PriorType: Prior>
   GaussianProcess_nalgebra<KernelType, PriorType>
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
      GaussianProcess_nalgebra { prior,
                        kernel,
                        noise,
                        training_inputs,
                        training_outputs,
                        covmat_cholesky}
   }
}

impl GaussianProcess_nalgebra<kernel::Gaussian, prior::Constant>
{
   /// returns a default gaussian process with a gaussian kernel and a constant prior, both fitted to the data
   pub fn default(training_inputs: DMatrix<f64>, training_outputs: DVector<f64>) -> Self
   {
      GaussianProcessBuilder_nalgebra::<kernel::Gaussian, prior::Constant>::new(training_inputs, training_outputs)
      .fit_kernel()
      .fit_prior()
      .train()
   }

   /// returns a default gaussian process with a gaussian kernel and a constant prior, both fitted to the data
   pub fn builder(
      training_inputs: DMatrix<f64>,
      training_outputs: DVector<f64>)
      -> GaussianProcessBuilder_nalgebra<kernel::Gaussian, prior::Constant>
   {
      GaussianProcessBuilder_nalgebra::<kernel::Gaussian, prior::Constant>::new(training_inputs,
                                                                                            training_outputs)
   }
}
