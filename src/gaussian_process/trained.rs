//! Trained Gaussian process

use nalgebra::{DMatrix};
use crate::parameters::kernel::Kernel;
use crate::parameters::prior::Prior;

/// gaussian process
pub struct GaussianProcessTrained<KernelType: Kernel, PriorType: Prior>
{
   /// value to which the process will regress in the absence of informations
   prior: PriorType,
   /// kernel used to fit the process on the data
   kernel: KernelType,
   /// amplitude of the noise of the data
   noise: f64,
   /// data used for fit
   training_inputs: DMatrix<f64>,
   training_outputs: DMatrix<f64>,
   /// cholesky decomposition of the covariance matrix trained on the current datapoints
   covariance_matrix_cholesky: DMatrix<f64>
}

impl<KernelType: Kernel, PriorType: Prior> GaussianProcessTrained<KernelType, PriorType>
{
   pub fn new(prior: PriorType,
              kernel: KernelType,
              noise: f64,
              training_inputs: DMatrix<f64>,
              training_outputs: DMatrix<f64>)
              -> Self
   {
      let covariance_matrix_cholesky = DMatrix::zeros(0, 0); // TODO
      GaussianProcessTrained::<KernelType, PriorType> { prior,
                                                        kernel,
                                                        noise,
                                                        training_inputs,
                                                        training_outputs,
                                                        covariance_matrix_cholesky };
      unimplemented!()
   }
   //----------------------------------------------------------------------------------------------
   // TRAINING

   /// adds new samples to the model
   /// update the model (which is faster than a training from scratch)
   /// does not refit the parameters
   pub fn add_samples(&mut self, inputs: DMatrix<f64>, outputs: DMatrix<f64>)
   {
      // TODO
      unimplemented!("update cholesky matrix")
   }

   /// fits the parameters and retrain the model from scratch
   pub fn fit_parameters(&mut self, fit_prior: bool, fit_kernel: bool)
   {
      if fit_prior
      {
         // TODO this needs to be fit on outputs with prior included and not deduced
         self.prior.fit(&self.training_inputs, &self.training_outputs);
      }

      if fit_kernel
      {
         self.kernel.fit(&self.training_inputs, &self.training_outputs);
      }

      unimplemented!("recompute cholesky matrix")
   }

   /// adds new samples to the model and fit the parameters
   /// faster than doing add_samples().fit_parameters()
   pub fn add_samples_fit(&mut self,
                          inputs: DMatrix<f64>,
                          outputs: DMatrix<f64>,
                          fit_prior: bool,
                          fit_kernel: bool)
   {
      // TODO add the samples

      self.fit_parameters(fit_prior, fit_kernel);
   }

   //----------------------------------------------------------------------------------------------
   // PREDICTION

   pub fn predict(&mut self, inputs: DMatrix<f64>) -> DMatrix<f64>
   {
      // TODO
      unimplemented!()
   }
}
