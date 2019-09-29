
use std::marker::PhantomData;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::StandardNormal;
use crate::conversion::AsVector;

/// modelize a multivariate normal distribution
pub struct MultivariateNormal<OutVector: AsVector>
{
   mean: DVector<f64>,
   cholesky_covariance: DMatrix<f64>,
   output_type: PhantomData<OutVector>
}

impl<OutVector: AsVector> MultivariateNormal<OutVector>
{
   /// outputs a new multivariate guassian with the given parameters
   pub fn new(mean: DVector<f64>, covariance: DMatrix<f64>) -> Self
   {
      let cholesky_covariance = covariance.cholesky().expect("Cholesky decomposition failed!").unpack();
      MultivariateNormal { mean, cholesky_covariance, output_type:PhantomData }
   }

   /// outputs the mean of the distribution
   pub fn mean(&self) -> OutVector
   {
      OutVector::from_vector(self.mean.clone())
   }

   /// takes a random number generator and uses it to sample from the distribution
   pub fn sample<RNG: Rng>(&self, rng: &mut RNG) -> OutVector
   {
      let normal = DVector::from_fn(self.mean.nrows(), |_, _| rng.sample(StandardNormal));
      let sample = &self.mean + &self.cholesky_covariance * normal;
      OutVector::from_vector(sample)
   }
}
