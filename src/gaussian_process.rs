//! Gaussian process

use nalgebra::{DVector, DMatrix};
use crate::kernel::{Kernel, Gaussian};

/// gaussian process
pub struct GaussianProcess<KernelType: Kernel>
{
   /// value to which the process will regress in the absence of informations
   prior: DVector<f64>, // TODO this could be a function, the mean or somethnig that is fitable like the prior
   /// kernel used to fit the process on the data
   kernel: KernelType,
   /// amplitude of the noise of the data
   noise: f64,
   training_inputs: DMatrix<f64>,
   training_outputs: DMatrix<f64>,
   /// cholesky decomposition of the covariance matrix trained on the current datapoints
   covariance_matrix_cholesky: DMatrix<f64>
}

impl<KernelType: Kernel> GaussianProcess<KernelType>
{
   /// builds a new gaussian process with default parameters
   /// the defaults are :
   /// - prior to 0
   /// - a gaussian kernel
   /// - a noise of 1e-7
   fn new(training_inputs: DMatrix<f64>, training_outputs: DMatrix<f64>) -> Self
   {
      let output_dimension = training_outputs.ncols();
      let prior = DVector::zeros(output_dimension);
      let kernel = KernelType::default();
      let noise = 1e-7f64;
      let nb_samples = training_inputs.nrows();
      let covariance_matrix_cholesky = DMatrix::zeros(nb_samples, nb_samples); // TODO
      GaussianProcess { prior, kernel, noise, training_inputs, training_outputs, covariance_matrix_cholesky }
   }

   //----------------------------------------------------------------------------------------------
   // PARAMETERS

   /// sets a new prior
   /// the prior is the value returned in the absence of information
   fn set_prior(&mut self, prior: DVector<f64>)
   {
      assert_eq!(self.prior.len(), prior.len());
      self.prior = prior;
      // TODO retrain
   }

   /// sets the noise parameters which correspond to the magnitude of the noise in the data
   fn set_noise(&mut self, noise: f64)
   {
      self.noise = noise;
      // TODO retrain
   }

   /// changes the kernel of the gaussian process
   fn set_kernel(&mut self, kernel: KernelType)
   {
      self.kernel = kernel;
      // TODO retrain
   }

   /// fits the parameters of the kernel on the training data
   fn fit_parameters(&mut self)
   {
      self.kernel.fit(&self.training_inputs, &self.training_outputs);
      // TODO retrain kernel
   }

   //----------------------------------------------------------------------------------------------
   // TRAINING

   /// adds new samples to the model
   fn add_samples(&mut self, inputs: DMatrix<f64>, outputs: DMatrix<f64>)
   {
      // TODO
      unimplemented!("update cholesky matrix")
   }

   //----------------------------------------------------------------------------------------------
   // PREDICTION

   fn predict(&mut self, inputs: DMatrix<f64>) -> DMatrix<f64>
   {
      // TODO
      unimplemented!()
   }
}
