use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::StandardNormal;

/// modelize a multidimensional multivariate normal distribution
/// multidimensional because each sample is a matrix and not just a vector
pub struct MultivariateNormal
{
   mean: DVector<f64>,
   cholesky_cov_inputs: DMatrix<f64>,
}

impl MultivariateNormal
{
   /// outputs a new multivariate guassian with the given parameters
   pub fn new(mean: DVector<f64>,
              covariance_inputs: DMatrix<f64>)
              -> Self
   {
      // covariance between the input points
      let cholesky_cov_inputs = covariance_inputs.cholesky().expect("Cholesky decomposition failed!").unpack();
      MultivariateNormal { mean, cholesky_cov_inputs }
   }

   /// outputs the mans of the distribution
   pub fn mean(&self) -> &DVector<f64>
   {
      &self.mean
   }

   /// takes a random number generator and uses it to sample from the distribution
   pub fn sample<RNG: Rng>(&self, rng: &mut RNG) -> DVector<f64>
   {
      // TODO can we just generate a vector from a fucntion ?
      let normals = DMatrix::from_fn(self.mean.nrows(), 1, |_, _| rng.sample(StandardNormal));
      // TODO is there a blas operation to make this operation faster ?
      &self.mean + &self.cholesky_cov_inputs * normals
   }
}
