//! Gaussian process builder

use nalgebra::{DMatrix};
use crate::kernel::{Kernel, Gaussian};
use crate::prior::{Prior, Constant};
use crate::gaussian_process::GaussianProcess;

/// gaussian process
/// TODO we would like this to not be tied to types until the prior has been chosen
pub struct GaussianProcessBuilder<KernelType: Kernel, PriorType: Prior>
{
   /// value to which the process will regress in the absence of informations
   prior: PriorType,
   /// kernel used to fit the process on the data
   kernel: KernelType,
   /// amplitude of the noise of the data
   noise: f64,
   training_inputs: DMatrix<f64>,
   training_outputs: DMatrix<f64>
}

impl<KernelType: Kernel, PriorType: Prior> GaussianProcessBuilder<KernelType, PriorType>
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
      // TODO are we using default kernels ?
      let output_dimension = training_outputs.ncols();
      let prior = Constant::default(output_dimension);
      let kernel = Gaussian::default();
      let noise = 1e-7f64;
      GaussianProcessBuilder { prior, kernel, noise, training_inputs, training_outputs }
   }

   //----------------------------------------------------------------------------------------------
   // SETTER

   /// sets a new prior
   /// the prior is the value returned in the absence of information
   pub fn set_prior<PriorType2: Prior>(self,
                                       prior: PriorType2)
                                       -> GaussianProcessBuilder<KernelType, PriorType2>
   {
      GaussianProcessBuilder { prior,
                               kernel: self.kernel,
                               noise: self.noise,
                               training_inputs: self.training_inputs,
                               training_outputs: self.training_outputs }
   }

   /// sets the noise parameters which correspond to the magnitude of the noise in the data
   pub fn set_noise(self, noise: f64) -> Self
   {
      GaussianProcessBuilder { noise, ..self }
   }

   /// changes the kernel of the gaussian process
   pub fn set_kernel<KernelType2: Kernel>(self,
                                          kernel: KernelType2)
                                          -> GaussianProcessBuilder<KernelType2, PriorType>
   {
      GaussianProcessBuilder { prior: self.prior,
                               kernel,
                               noise: self.noise,
                               training_inputs: self.training_inputs,
                               training_outputs: self.training_outputs }
   }

   //----------------------------------------------------------------------------------------------
   // FIT

   /// fits the parameters of the kernel on the training data
   pub fn fit_parameters(mut self) -> Self
   {
      self.kernel.fit(&self.training_inputs, &self.training_outputs);
      self
   }

   /// fits the prior on the training data
   pub fn fit_prior(mut self) -> Self
   {
      self.prior.fit(&self.training_inputs, &self.training_outputs);
      self
   }

   //----------------------------------------------------------------------------------------------
   // TRAIN

   /// trains the gaussian process
   pub fn train(self) -> GaussianProcess<KernelType, PriorType>
   {
      // TODO
      unimplemented!()
   }
}
