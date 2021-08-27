//! Gaussian process
//!
//! ```rust
//! # use friedrich::gaussian_process::GaussianProcess;
//! # fn main() {
//! // trains a gaussian process on a dataset of one dimension vectors
//! let training_inputs = vec![vec![0.8], vec![1.2], vec![3.8], vec![4.2]];
//! let training_outputs = vec![3.0, 4.0, -2.0, -2.0];
//! let mut gp = GaussianProcess::default(training_inputs, training_outputs);
//!
//! // predicts the mean and variance of a single point
//! let input = vec![1.];
//! let mean = gp.predict(&input);
//! let var = gp.predict_variance(&input);
//! println!("prediction: {} ± {}", mean, var.sqrt());
//!
//! // makes several prediction
//! let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
//! let outputs = gp.predict(&inputs);
//! println!("predictions: {:?}", outputs);
//!
//! // updates the model
//! let additional_inputs = vec![vec![0.], vec![1.], vec![2.], vec![5.]];
//! let additional_outputs = vec![2.0, 3.0, -1.0, -2.0];
//! let fit_prior = true;
//! let fit_kernel = true;
//! let max_iter = 100;
//! let convergence_fraction = 0.05;
//! gp.add_samples(&additional_inputs, &additional_outputs);
//! gp.fit_parameters(fit_prior, fit_kernel, max_iter, convergence_fraction);
//!
//! // samples from the distribution
//! let new_inputs = vec![vec![1.0], vec![2.0]];
//! let sampler = gp.sample_at(&new_inputs);
//! let mut rng = rand::thread_rng();
//! for i in 1..=5
//! {
//!   println!("sample {} : {:?}", i, sampler.sample(&mut rng));
//! }
//! # }
//! ```

use crate::algebra::{
    add_rows_cholesky_cov_matrix, make_cholesky_cov_matrix, make_covariance_matrix, EMatrix,
    EVector,
};
use crate::conversion::Input;
use crate::parameters::{kernel, kernel::Kernel, prior, prior::Prior};
use nalgebra::{Cholesky, DMatrix, DVector, Dynamic};

mod multivariate_normal;
pub use multivariate_normal::MultivariateNormal;

mod builder;
pub use builder::GaussianProcessBuilder;

mod optimizer;

/// A Gaussian process that can be used to make predictions based on its training data
pub struct GaussianProcess<KernelType: Kernel, PriorType: Prior> {
    /// value to which the process will regress in the absence of informations
    pub prior: PriorType,
    /// kernel used to fit the process on the data
    pub kernel: KernelType,
    /// amplitude of the noise of the data as provided by the user or deduced by the optimizer
    pub noise: f64,
    /// data used for fit
    training_inputs: EMatrix,
    training_outputs: EVector,
    /// cholesky decomposition of the covariance matrix trained on the current datapoints
    covmat_cholesky: Cholesky<f64, Dynamic>,
}

impl GaussianProcess<kernel::Gaussian, prior::ConstantPrior> {
    /// Returns a gaussian process with a Gaussian kernel and a constant prior, both fitted to the data.
    ///
    /// ```rust
    /// # use friedrich::gaussian_process::GaussianProcess;
    /// # fn main() {
    /// // training data
    /// let training_inputs = vec![vec![0.8], vec![1.2], vec![3.8], vec![4.2]];
    /// let training_outputs = vec![3.0, 4.0, -2.0, -2.0];
    ///
    /// // defining and training a model
    /// let gp = GaussianProcess::default(training_inputs, training_outputs);
    /// # }
    /// ```
    pub fn default<T: Input>(training_inputs: T, training_outputs: T::InVector) -> Self {
        GaussianProcessBuilder::<kernel::Gaussian, prior::ConstantPrior>::new(
            training_inputs,
            training_outputs,
        )
        .fit_kernel()
        .fit_prior()
        .train()
    }

    /// Returns a builder to define specific parameters of the gaussian process.
    ///
    /// ```rust
    /// # use friedrich::gaussian_process::GaussianProcess;
    /// # use friedrich::prior::*;
    /// # use friedrich::kernel::*;
    /// # fn main() {
    /// # // training data
    /// # let training_inputs = vec![vec![0.8], vec![1.2], vec![3.8], vec![4.2]];
    /// # let training_outputs = vec![3.0, 4.0, -2.0, -2.0];
    /// // model parameters
    /// let input_dimension = 1;
    /// let output_noise = 0.1;
    /// let exponential_kernel = Exponential::default();
    /// let linear_prior = LinearPrior::default(input_dimension);
    ///
    /// // defining and training a model
    /// let gp = GaussianProcess::builder(training_inputs, training_outputs).set_noise(output_noise)
    ///                                                                     .set_kernel(exponential_kernel)
    ///                                                                     .fit_kernel()
    ///                                                                     .set_prior(linear_prior)
    ///                                                                     .fit_prior()
    ///                                                                     .train();
    /// # }
    /// ```
    pub fn builder<T: Input>(
        training_inputs: T,
        training_outputs: T::InVector,
    ) -> GaussianProcessBuilder<kernel::Gaussian, prior::ConstantPrior> {
        GaussianProcessBuilder::<kernel::Gaussian, prior::ConstantPrior>::new(
            training_inputs,
            training_outputs,
        )
    }
}

impl<KernelType: Kernel, PriorType: Prior> GaussianProcess<KernelType, PriorType> {
    /// Raw method to create a new gaussian process with the given parameters / data.
    /// We recommend that you use either the default parameters or the builder to simplify the definition process.
    pub fn new<T: Input>(
        prior: PriorType,
        kernel: KernelType,
        noise: f64,
        training_inputs: T,
        training_outputs: T::InVector,
    ) -> Self {
        assert!(
            noise >= 0.,
            "The noise parameter should non-negative but we tried to set it to {}",
            noise
        );
        let training_inputs = T::into_dmatrix(training_inputs);
        let training_outputs = T::into_dvector(training_outputs);
        assert_eq!(training_inputs.nrows(), training_outputs.nrows());
        // converts training data into extendable matrix
        let training_inputs = EMatrix::new(training_inputs);
        let training_outputs =
            EVector::new(training_outputs - prior.prior(&training_inputs.as_matrix()));
        // computes cholesky decomposition
        let covmat_cholesky =
            make_cholesky_cov_matrix(&training_inputs.as_matrix(), &kernel, noise);
        GaussianProcess {
            prior,
            kernel,
            noise,
            training_inputs,
            training_outputs,
            covmat_cholesky,
        }
    }

    /// Adds new samples to the model.
    ///
    /// Updates the model (which is faster than a retraining from scratch)
    /// but does not refit the parameters.
    pub fn add_samples<T: Input>(&mut self, inputs: &T, outputs: &T::InVector) {
        let inputs = T::to_dmatrix(inputs);
        let outputs = T::to_dvector(outputs);
        assert_eq!(inputs.nrows(), outputs.nrows());
        assert_eq!(inputs.ncols(), self.training_inputs.as_matrix().ncols());
        // grows the training matrix
        let outputs = outputs - self.prior.prior(&inputs);
        self.training_inputs.add_rows(&inputs);
        self.training_outputs.add_rows(&outputs);
        // add new rows to cholesky matrix
        let nb_new_inputs = inputs.nrows();
        add_rows_cholesky_cov_matrix(
            &mut self.covmat_cholesky,
            &self.training_inputs.as_matrix(),
            nb_new_inputs,
            &self.kernel,
            self.noise,
        );
    }

    /// Computes the log likelihood of the current model given the training data.
    ///
    /// This quantity can be used for model selection.
    /// Given two models, the one with the highest score would be the one with the highest probability of producing the data.
    pub fn likelihood(&self) -> f64 {
        // formula : -1/2 (transpose(output)*cov(train,train)^-1*output + trace(log|cov(train,train)|) + size(train)*log(2*pi))

        // how well do we fit the trainnig data ?
        let output = self.training_outputs.as_vector().clone();
        // transpose(ol)*ol = transpose(output)*cov(train,train)^-1*output
        let ol = self
            .covmat_cholesky
            .l()
            .solve_lower_triangular(&output)
            .expect("likelihood : solve failed");
        let data_fit: f64 = ol.norm_squared();

        // penalizes complex models
        // recomputing kernels seems easier than extracting and squaring the diagonal of the cholesky matrix
        let complexity_penalty: f64 = self
            .training_inputs
            .as_matrix()
            .row_iter()
            .map(|r| self.kernel.kernel(&r, &r) + self.noise * self.noise)
            .map(|c| c.abs().ln())
            .sum();

        // rescales the output to make it independant of the number of samples
        let n = self.training_inputs.as_matrix().nrows();
        let normalization_constant = (n as f64) * (2. * std::f64::consts::PI).ln();

        -(data_fit + complexity_penalty + normalization_constant) / 2.
    }

    //----------------------------------------------------------------------------------------------
    // PREDICT

    /// Makes a prediction (the mean of the gaussian process) for each row of the input.
    pub fn predict<T: Input>(&self, inputs: &T) -> T::OutVector {
        // formula : prior + cov(input,train)*cov(train,train)^-1 * output

        let inputs = T::to_dmatrix(inputs);
        assert_eq!(inputs.ncols(), self.training_inputs.as_matrix().ncols());

        // computes weights to give each training sample
        let mut weights =
            make_covariance_matrix(&self.training_inputs.as_matrix(), &inputs, &self.kernel);
        self.covmat_cholesky.solve_mut(&mut weights);

        // computes prior for the given inputs
        let mut prior = self.prior.prior(&inputs);

        // weights.transpose() * &self.training_outputs + prior
        prior.gemm_tr(1f64, &weights, &self.training_outputs.as_vector(), 1f64);

        T::from_dvector(&prior)
    }

    /// Predicts the variance of the gaussian process for each row of the input.
    /// This quantity (and its square root) can be used as a proxy for the uncertainty of the prediction.
    pub fn predict_variance<T: Input>(&self, inputs: &T) -> T::OutVector {
        // formula, diagonal of : cov(input,input) - cov(input,train)*cov(train,train)^-1*cov(train,input)

        let inputs = T::to_dmatrix(inputs);
        assert_eq!(inputs.ncols(), self.training_inputs.as_matrix().ncols());

        // compute the covariances
        let cov_train_inputs =
            make_covariance_matrix(&self.training_inputs.as_matrix(), &inputs, &self.kernel);

        // solve linear system
        let kl = self
            .covmat_cholesky
            .l()
            .solve_lower_triangular(&cov_train_inputs)
            .expect("predict_covariance : solve failed");

        // (cov_inputs_inputs - (kl.transpose() * kl)).diagonal()
        let variances = inputs
            .row_iter()
            .map(|row| self.kernel.kernel(&row, &row)) // variance of input points with themselves
            .zip(kl.column_iter().map(|col| col.norm_squared())) // diag(kl^T * kl)
            .map(|(base_cov, predicted_cov)| base_cov - predicted_cov);
        let variances = DVector::<f64>::from_iterator(inputs.nrows(), variances);

        T::from_dvector(&variances)
    }

    /// Predicts both the mean and the variance of the gaussian process for each row of the input.
    ///
    /// Faster than calling `predict` and `predict_variance` separately.
    ///
    /// ```rust
    /// # use friedrich::gaussian_process::GaussianProcess;
    /// # fn main() {
    /// # let training_inputs = vec![vec![0.8], vec![1.2], vec![3.8], vec![4.2]];
    /// # let training_outputs = vec![3.0, 4.0, -2.0, -2.0];
    /// let gp = GaussianProcess::default(training_inputs, training_outputs);
    /// let input = vec![1.];
    /// let (mean, var) = gp.predict_mean_variance(&input);
    /// println!("prediction: {} ± {}", mean, var.sqrt());
    /// # }
    /// ```
    pub fn predict_mean_variance<T: Input>(&self, inputs: &T) -> (T::OutVector, T::OutVector) {
        let inputs = T::to_dmatrix(inputs);
        assert_eq!(inputs.ncols(), self.training_inputs.as_matrix().ncols());

        // computes weights to give each training sample
        let cov_train_inputs =
            make_covariance_matrix(&self.training_inputs.as_matrix(), &inputs, &self.kernel);
        let weights = self.covmat_cholesky.solve(&cov_train_inputs);

        // ----- mean -----

        // computes prior for the given inputs
        let mut prior = self.prior.prior(&inputs);

        // weights.transpose() * &self.training_outputs + prior
        prior.gemm_tr(1f64, &weights, &self.training_outputs.as_vector(), 1f64);

        let mean = T::from_dvector(&prior);

        // ----- variance -----

        // (cov_inputs_inputs - cov_train_inputs.transpose() * weights).diagonal()
        let mut variances = DVector::<f64>::zeros(inputs.nrows());
        for (i, input) in inputs.row_iter().enumerate() {
            let base_cov = self.kernel.kernel(&input, &input);
            let predicted_cov = cov_train_inputs.column(i).dot(&weights.column(i));
            variances[i] = base_cov - predicted_cov;
        }

        let variance = T::from_dvector(&variances);

        // ----- result -----

        (mean, variance)
    }

    /// Returns the covariance matrix for the rows of the input.
    pub fn predict_covariance<T: Input>(&self, inputs: &T) -> DMatrix<f64> {
        // formula : cov(input,input) - cov(input,train)*cov(train,train)^-1*cov(train,input)

        let inputs = T::to_dmatrix(inputs);
        assert_eq!(inputs.ncols(), self.training_inputs.as_matrix().ncols());

        // compute the covariances
        let cov_train_inputs =
            make_covariance_matrix(&self.training_inputs.as_matrix(), &inputs, &self.kernel);
        let mut cov_inputs_inputs = make_covariance_matrix(&inputs, &inputs, &self.kernel);

        // solve linear system
        let kl = self
            .covmat_cholesky
            .l()
            .solve_lower_triangular(&cov_train_inputs)
            .expect("predict_covariance : solve failed");

        // cov_inputs_inputs - (kl.transpose() * kl)
        cov_inputs_inputs.gemm_tr(-1f64, &kl, &kl, 1f64);
        cov_inputs_inputs
    }

    /// Produces a multivariate gaussian that can be used to sample at the input points.
    ///
    /// The sampling requires a random number generator compatible with the [rand](https://crates.io/crates/rand) crate :
    ///
    /// ```rust
    /// # use friedrich::gaussian_process::GaussianProcess;
    /// # fn main() {
    /// # let training_inputs = vec![vec![0.8], vec![1.2], vec![3.8], vec![4.2]];
    /// # let training_outputs = vec![3.0, 4.0, -2.0, -2.0];
    /// # let gp = GaussianProcess::default(training_inputs, training_outputs);
    /// // computes the distribution at some new coordinates
    /// let new_inputs = vec![vec![1.], vec![2.]];
    /// let sampler = gp.sample_at(&new_inputs);
    ///
    /// // samples from the distribution
    /// let mut rng = rand::thread_rng();
    /// println!("samples a vector : {:?}", sampler.sample(&mut rng));
    /// # }
    /// ```
    pub fn sample_at<T: Input>(&self, inputs: &T) -> MultivariateNormal<T> {
        let inputs = T::to_dmatrix(inputs);
        assert_eq!(inputs.ncols(), self.training_inputs.as_matrix().ncols());

        // compute the weights
        let cov_train_inputs =
            make_covariance_matrix(&self.training_inputs.as_matrix(), &inputs, &self.kernel);
        let weights = self.covmat_cholesky.solve(&cov_train_inputs);

        // computes covariance
        let mut cov_inputs_inputs = make_covariance_matrix(&inputs, &inputs, &self.kernel);
        cov_inputs_inputs.gemm_tr(-1f64, &cov_train_inputs, &weights, 1f64);
        let cov = cov_inputs_inputs;

        // computes the mean
        let mut prior = self.prior.prior(&inputs);
        prior.gemm_tr(1f64, &weights, &self.training_outputs.as_vector(), 1f64);
        let mean = prior;

        MultivariateNormal::new(mean, cov)
    }

    //----------------------------------------------------------------------------------------------
    // FIT

    /// Fits the requested parameters and retrains the model.
    ///
    /// The fit of the noise and kernel parameters is done by gradient descent.
    /// It runs for a maximum of `max_iter` iterations and stops prematurely if all gradients are below `convergence_fraction` time their associated parameter.
    ///
    /// Good base values for `max_iter` and `convergence_fraction` are 100 and 0.05
    ///
    /// Note that, if the `noise` parameter ends up unnaturaly large after the fit, it is a good sign that the kernel is unadapted to the data.
    pub fn fit_parameters(
        &mut self,
        fit_prior: bool,
        fit_kernel: bool,
        max_iter: usize,
        convergence_fraction: f64,
    ) {
        if fit_prior {
            // gets the original data back in order to update the prior
            let training_outputs = self.training_outputs.as_vector()
                + self.prior.prior(&self.training_inputs.as_matrix());
            self.prior
                .fit(&self.training_inputs.as_matrix(), &training_outputs);
            let training_outputs =
                training_outputs - self.prior.prior(&self.training_inputs.as_matrix());
            self.training_outputs.assign(&training_outputs);
            // NOTE: adding and substracting each time we fit a prior might be numerically unwise

            if !fit_kernel {
                // retrains model from scratch
                self.covmat_cholesky = make_cholesky_cov_matrix(
                    &self.training_inputs.as_matrix(),
                    &self.kernel,
                    self.noise,
                );
            }
        }

        // fit kernel and retrains model from scratch
        if fit_kernel {
            if self.kernel.is_scaleable() {
                self.scaled_optimize_parameters(max_iter, convergence_fraction);
            } else {
                self.optimize_parameters(max_iter, convergence_fraction);
            }
        }
    }
}
