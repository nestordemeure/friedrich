//! Methods to build a Gaussian process.

use super::GaussianProcess;
use super::builder::GaussianProcessBuilder;
use crate::conversion::{AsMatrix, AsVector};
use crate::parameters::{kernel::Kernel, prior::Prior};
use crate::parameters::*;
use crate::algebra::{EMatrix, EVector};
use std::marker::PhantomData;
use crate::algebra;

impl<KernelType: Kernel, PriorType: Prior, OutVector: AsVector>
   GaussianProcess<KernelType, PriorType, OutVector>
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
      GaussianProcess { prior,
                        kernel,
                        noise,
                        training_inputs,
                        training_outputs,
                        covmat_cholesky,
                        output_type: PhantomData }
   }
}

impl<OutVector: AsVector> GaussianProcess<kernel::Gaussian, prior::Constant, OutVector>
{
   /// returns a default gaussian process with a gaussian kernel and a constant prior, both fitted to the data
   pub fn default<InMatrix: AsMatrix>(training_inputs: InMatrix, training_outputs: OutVector) -> Self
   {
      GaussianProcessBuilder::<kernel::Gaussian, prior::Constant, InMatrix, OutVector>::new(training_inputs, training_outputs)
      .fit_kernel()
      .fit_prior()
      .train()
   }

   /// returns a default gaussian process with a gaussian kernel and a constant prior, both fitted to the data
   pub fn builder<InMatrix: AsMatrix>(
      training_inputs: InMatrix,
      training_outputs: OutVector)
      -> GaussianProcessBuilder<kernel::Gaussian, prior::Constant, InMatrix, OutVector>
   {
      GaussianProcessBuilder::<kernel::Gaussian, prior::Constant, InMatrix, OutVector>::new(training_inputs,
                                                                                            training_outputs)
   }
}
