//! Gaussian process

use nalgebra::{DMatrix};
use crate::kernel::{Kernel, Gaussian};
use crate::prior::{Prior, Constant};
use crate::gaussian_process_builder::GaussianProcessBuilder;

/// gaussian process
pub struct GaussianProcess<KernelType: Kernel, PriorType: Prior>
{
   /// value to which the process will regress in the absence of informations
   prior: PriorType, // TODO this could be a function, the mean or somethnig that is fitable like the prior
   /// kernel used to fit the process on the data
   kernel: KernelType,
   /// amplitude of the noise of the data
   noise: f64,
   training_inputs: DMatrix<f64>,
   training_outputs: DMatrix<f64>,
   /// cholesky decomposition of the covariance matrix trained on the current datapoints
   covariance_matrix_cholesky: DMatrix<f64>
}

impl<KernelType: Kernel, PriorType: Prior> GaussianProcess<KernelType, PriorType>
{
   /// builds a new gaussian process with default parameters
   /// the defaults are :
   /// - constant prior set to 0
   /// - a gaussian kernel
   /// - a noise of 1e-7
   pub fn new(training_inputs: DMatrix<f64>,
          training_outputs: DMatrix<f64>)
          -> GaussianProcessBuilder<Gaussian, Constant>
   {
      GaussianProcessBuilder::<KernelType, PriorType>::new(training_inputs, training_outputs)
   }

   //----------------------------------------------------------------------------------------------
   // FIT

   /// fits the parameters of the kernel on the training data
   pub fn fit_parameters(&mut self)
   {
      // TODO this can be fitted on output with prior deduced
      self.kernel.fit(&self.training_inputs, &self.training_outputs);
      // TODO retrain kernel
   }

   /// fits the prior on the training data
   pub fn fit_prior(&mut self)
   {
      // TODO this needs to be fit on outputs with prior included and not deduced
      self.prior.fit(&self.training_inputs, &self.training_outputs);
      // TODO retrain kernel
   }

   //----------------------------------------------------------------------------------------------
   // TRAINING

   /// adds new samples to the model
   pub fn add_samples(&mut self, inputs: DMatrix<f64>, outputs: DMatrix<f64>)
   {
      // TODO
      unimplemented!("update cholesky matrix")
   }

   //----------------------------------------------------------------------------------------------
   // PREDICTION

   pub fn predict(&mut self, inputs: DMatrix<f64>) -> DMatrix<f64>
   {
      // TODO
      unimplemented!()
   }
}
