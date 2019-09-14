use nalgebra::DMatrix;
use rand::Rng;
use rand_distr::StandardNormal;

/// modelize a multidimensional multivariate normal distribution
/// multidimensional because each sample is a matrix and not just a vector
pub struct MultivariateNormal
{
   mean: DMatrix<f64>,
   cholesky_cov_inputs: DMatrix<f64>,
   cholesky_cov_output_dim: DMatrix<f64>
}

impl MultivariateNormal
{
   /// outputs a new multivariate guassian with the given parameters
   pub fn new(mean: DMatrix<f64>,
              covariance_inputs: DMatrix<f64>,
              covariance_output_dim: DMatrix<f64>)
              -> Self
   {
      let cholesky_cov_inputs =
         covariance_inputs.cholesky().expect("Cholesky decomposition failed!").unpack();
      let cholesky_cov_output_dim =
         covariance_output_dim.cholesky().expect("Cholesky decomposition failed!").unpack().transpose();
      MultivariateNormal { mean, cholesky_cov_inputs, cholesky_cov_output_dim }
   }

   /// outputs the mans of the distribution
   pub fn mean(&self) -> &DMatrix<f64>
   {
      &self.mean
   }

   /// takes a random number generator and uses it to sample from the distribution
   pub fn sample<RNG: Rng>(&self, rng: &mut RNG) -> DMatrix<f64>
   {
      let normals = DMatrix::from_fn(self.mean.nrows(), self.mean.ncols(), |_, _| rng.sample(StandardNormal));
      &self.mean + &self.cholesky_cov_inputs * normals * &self.cholesky_cov_output_dim
   }
}
