//! Gaussian process builder

use nalgebra::{DMatrix, DVector};
use crate::input::{Input, Output};
use crate::parameters::kernel::Kernel;
use crate::parameters::prior::Prior;
use super::trained::GaussianProcessTrained;

/// gaussian process builder
/// used to define the paramters of the gaussian process
pub struct GaussianProcessBuilder<KernelType: Kernel, PriorType: Prior>
{
   /// value to which the process will regress in the absence of informations
   prior: PriorType,
   /// kernel used to fit the process on the data
   kernel: KernelType,
   /// amplitude of the noise of the data
   noise: f64,
   /// type of fit to be applied
   should_fit_kernel: bool,
   should_fit_prior: bool,
   /// data use for training
   training_inputs: DMatrix<f64>,
   training_outputs: DVector<f64>
}

impl<KernelType: Kernel, PriorType: Prior> GaussianProcessBuilder<KernelType, PriorType>
{
   /// builds a new gaussian process with default parameters
   /// the defaults are :
   /// - constant prior set to 0
   /// - a gaussian kernel
   /// - a noise of 1e-7
   /// - does not fit parameters
   /// - does fit prior
   pub fn new<InMatrix: Input, OutVector: Output>(training_inputs: InMatrix,
                                                  training_outputs: OutVector)
                                                  -> GaussianProcessBuilder<KernelType, PriorType>
   {
      let training_inputs = training_inputs.into_input();
      let training_outputs = training_outputs.into_output();
      let input_dimension = training_inputs.ncols();
      let prior = PriorType::default(input_dimension);
      let kernel = KernelType::default();
      let noise = 1e-7f64;
      let should_fit_kernel = false;
      let should_fit_prior = false;
      GaussianProcessBuilder { prior,
                               kernel,
                               noise,
                               should_fit_kernel,
                               should_fit_prior,
                               training_inputs,
                               training_outputs }
   }

   //----------------------------------------------------------------------------------------------
   // SETTERS

   /// sets a new prior
   /// the prior is the value returned in the absence of information
   pub fn set_prior<PriorType2: Prior>(self,
                                       prior: PriorType2)
                                       -> GaussianProcessBuilder<KernelType, PriorType2>
   {
      GaussianProcessBuilder { prior,
                               kernel: self.kernel,
                               noise: self.noise,
                               should_fit_kernel: self.should_fit_kernel,
                               should_fit_prior: self.should_fit_prior,
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
                               should_fit_kernel: self.should_fit_kernel,
                               should_fit_prior: self.should_fit_prior,
                               training_inputs: self.training_inputs,
                               training_outputs: self.training_outputs }
   }

   /// fits the parameters of the kernel on the training data
   pub fn fit_kernel(self) -> Self
   {
      GaussianProcessBuilder { should_fit_kernel: true, ..self }
   }

   /// fits the prior on the training data
   pub fn fit_prior(self) -> Self
   {
      GaussianProcessBuilder { should_fit_prior: true, ..self }
   }

   //----------------------------------------------------------------------------------------------
   // TRAIN

   /// trains the gaussian process
   pub fn train(self) -> GaussianProcessTrained<KernelType, PriorType>
   {
      // builds a gp with no data
      let empty_input = DMatrix::zeros(0, self.training_inputs.ncols());
      let empty_output = DVector::zeros(0);
      let mut gp = GaussianProcessTrained::<KernelType, PriorType>::new(self.prior,
                                                                        self.kernel,
                                                                        self.noise,
                                                                        empty_input,
                                                                        empty_output);
      // trains it (and fits it if requested) on the training data
      gp.add_samples_fit(self.training_inputs,
                         self.training_outputs,
                         self.should_fit_prior,
                         self.should_fit_kernel);
      gp
   }
}
