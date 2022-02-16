use crate::conversion::Input;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::StandardNormal;
use std::marker::PhantomData;

/// Multivariate Normal Distribution
///
/// This class is meant to be produced by the `sample_at` method of the gaussian process
/// and can be used to sample the process at a given point / set of points :
///
/// ```rust
/// # use friedrich::gaussian_process::GaussianProcess;
/// # fn main() {
/// // trains a model
/// let training_inputs = vec![vec![0.8], vec![1.2], vec![3.8], vec![4.2]];
/// let training_outputs = vec![3.0, 4.0, -2.0, -2.0];
/// let gp = GaussianProcess::default(training_inputs, training_outputs);
///
/// // computes the distribution at some new coordinates
/// let new_inputs = vec![vec![1.], vec![2.]];
/// let sampler = gp.sample_at(&new_inputs);
///
/// // samples from the distribution
/// let mut rng = rand::thread_rng();
/// println!("samples a vector : {:?}", sampler.sample(&mut rng));
/// # }
/// ```
///
/// Note that the output type is a function of the input of `sample_at`, the method can be used on a vector of vectors as well as a single row :
///
/// ```rust
/// # use friedrich::gaussian_process::GaussianProcess;
/// # fn main() {
/// # // trains a model
/// # let training_inputs = vec![vec![0.8], vec![1.2], vec![3.8], vec![4.2]];
/// # let training_outputs = vec![3.0, 4.0, -2.0, -2.0];
/// # let gp = GaussianProcess::default(training_inputs, training_outputs);
/// // computes the distribution at some new coordinate
/// let new_input = vec![1.];
/// let sampler = gp.sample_at(&new_input);
///
/// // samples from the distribution
/// let mut rng = rand::thread_rng();
/// println!("samples a value : {}", sampler.sample(&mut rng));
/// # }
/// ```
pub struct MultivariateNormal<T: Input> {
    mean: DVector<f64>,
    cholesky_covariance: DMatrix<f64>,
    input_type: PhantomData<T>,
}

impl<T: Input> MultivariateNormal<T> {
    /// Produces a new multivariate guassian with the given parameters
    pub fn new(mean: DVector<f64>, covariance: DMatrix<f64>) -> Self {
        let cholesky_covariance = covariance
            .cholesky()
            .expect("MultivariateNormal: Cholesky decomposition failed!")
            .unpack();
        MultivariateNormal {
            mean,
            cholesky_covariance,
            input_type: PhantomData,
        }
    }

    /// Outputs the mean of the distribution
    pub fn mean(&self) -> T::OutVector {
        T::from_dvector(&self.mean)
    }

    /// Takes a random number generator and uses it to sample from the distribution
    pub fn sample<RNG: Rng>(&self, rng: &mut RNG) -> T::OutVector {
        let normal = DVector::from_fn(self.mean.nrows(), |_, _| rng.sample(StandardNormal));
        let sample = &self.mean + &self.cholesky_covariance * normal;
        T::from_dvector(&sample)
    }
}
