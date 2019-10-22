
use rand::Rng;
use rand_distr::StandardNormal;
use nalgebra::{DMatrix, DVector};

/// modelize a multivariate normal distribution
pub struct MultivariateNormal
{
   mean: DVector<f64>,
   cholesky_covariance: DMatrix<f64>
}

impl MultivariateNormal
{
   /// outputs a new multivariate guassian with the given parameters
   pub fn new(mean: DVector<f64>, covariance: DMatrix<f64>) -> Self
   {
      let cholesky_covariance = covariance.cholesky().expect("Cholesky decomposition failed!").unpack();
      MultivariateNormal { mean, cholesky_covariance }
   }

   /// outputs the mean of the distribution
   pub fn mean(&self) -> &DVector<f64>
   {
      &self.mean
   }

   /// takes a random number generator and uses it to sample from the distribution
   pub fn sample<RNG: Rng>(&self, rng: &mut RNG) -> DVector<f64>
   {
      let normal = DVector::from_fn(self.mean.nrows(), |_, _| rng.sample(StandardNormal));
      let sample = &self.mean + &self.cholesky_covariance * normal;
      sample
   }
}
