use super::GaussianProcess;
use crate::conversion::Input;
use crate::parameters::kernel::Kernel;
use crate::parameters::prior::Prior;
use nalgebra::{DMatrix, DVector};

/// Builder to set the parameters of a gaussian process.
///
/// This class is meant to be produced by the `builder` method of the gaussian process and can be used to select the various parameters of the gaussian process :
///
/// ```rust
/// # use friedrich::gaussian_process::GaussianProcess;
/// # use friedrich::prior::*;
/// # use friedrich::kernel::*;
/// # fn main() {
/// // training data
/// let training_inputs = vec![vec![0.8], vec![1.2], vec![3.8], vec![4.2]];
/// let training_outputs = vec![3.0, 4.0, -2.0, -2.0];
///
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
pub struct GaussianProcessBuilder<KernelType: Kernel, PriorType: Prior> {
    /// value to which the process will regress in the absence of informations
    prior: PriorType,
    /// kernel used to fit the process on the data
    kernel: KernelType,
    /// amplitude of the noise of the data
    noise: f64,
    /// type of fit to be applied
    should_fit_kernel: bool,
    should_fit_prior: bool,
    /// fit parameters
    max_iter: usize,
    convergence_fraction: f64,
    /// data use for training
    training_inputs: DMatrix<f64>,
    training_outputs: DVector<f64>,
}

impl<KernelType: Kernel, PriorType: Prior> GaussianProcessBuilder<KernelType, PriorType> {
    /// builds a new gaussian process with default parameters
    ///
    /// the defaults are :
    /// - constant prior (0 unless fitted)
    /// - a gaussian kernel
    /// - a noise of 10% of the output standard deviation (might be re-fitted in the absence of user provided value)
    /// - does not fit parameters
    pub fn new<T: Input>(training_inputs: T, training_outputs: T::InVector) -> Self {
        let training_inputs = T::into_dmatrix(training_inputs);
        let training_outputs = T::into_dvector(training_outputs);
        // makes builder
        let prior = PriorType::default(training_inputs.ncols());
        let kernel = KernelType::default();
        let noise = 0.1 * training_outputs.row_variance()[0].sqrt(); // 10% of output std by default
        let should_fit_kernel = false;
        let should_fit_prior = false;
        let max_iter = 100;
        let convergence_fraction = 0.05;
        GaussianProcessBuilder {
            prior,
            kernel,
            noise,
            should_fit_kernel,
            should_fit_prior,
            max_iter,
            convergence_fraction,
            training_inputs,
            training_outputs,
        }
    }

    //----------------------------------------------------------------------------------------------
    // SETTERS

    /// Sets a new prior.
    /// See the documentation on priors for more informations.
    pub fn set_prior<NewPriorType: Prior>(
        self,
        prior: NewPriorType,
    ) -> GaussianProcessBuilder<KernelType, NewPriorType> {
        GaussianProcessBuilder {
            prior,
            kernel: self.kernel,
            noise: self.noise,
            should_fit_kernel: self.should_fit_kernel,
            should_fit_prior: self.should_fit_prior,
            max_iter: self.max_iter,
            convergence_fraction: self.convergence_fraction,
            training_inputs: self.training_inputs,
            training_outputs: self.training_outputs,
        }
    }

    /// Sets the noise parameter.
    /// It correspond to the standard deviation of the noise in the outputs of the training set.
    pub fn set_noise(self, noise: f64) -> Self {
        assert!(
            noise >= 0.,
            "The noise parameter should non-negative but we tried to set it to {}",
            noise
        );
        GaussianProcessBuilder { noise, ..self }
    }

    /// Changes the kernel of the gaussian process.
    /// See the documentations on Kernels for more informations.
    pub fn set_kernel<NewKernelType: Kernel>(
        self,
        kernel: NewKernelType,
    ) -> GaussianProcessBuilder<NewKernelType, PriorType> {
        GaussianProcessBuilder {
            prior: self.prior,
            kernel,
            noise: self.noise,
            should_fit_kernel: self.should_fit_kernel,
            should_fit_prior: self.should_fit_prior,
            max_iter: self.max_iter,
            convergence_fraction: self.convergence_fraction,
            training_inputs: self.training_inputs,
            training_outputs: self.training_outputs,
        }
    }

    /// Modifies the stopping criteria of the gradient descent used to fit the noise and kernel parameters.
    ///
    /// The optimizer runs for a maximum of `max_iter` iterations and stops prematurely if all gradients are below `convergence_fraction` time their associated parameter.
    pub fn set_fit_parameters(self, max_iter: usize, convergence_fraction: f64) -> Self {
        GaussianProcessBuilder {
            max_iter,
            convergence_fraction,
            ..self
        }
    }

    /// Asks for the parameters of the kernel to be fitted on the training data.
    /// The fitting will be done when the `train` method is called.
    pub fn fit_kernel(self) -> Self {
        GaussianProcessBuilder {
            should_fit_kernel: true,
            ..self
        }
    }

    /// Asks for the prior to be fitted on the training data.
    /// The fitting will be done when the `train` method is called.
    pub fn fit_prior(self) -> Self {
        GaussianProcessBuilder {
            should_fit_prior: true,
            ..self
        }
    }

    //----------------------------------------------------------------------------------------------
    // TRAIN

    /// Trains the gaussian process.
    /// Fits the parameters if requested.
    pub fn train(mut self) -> GaussianProcess<KernelType, PriorType> {
        // prepare kernel and noise values using heuristics
        // TODO how to detect if values have been entered by the user meaning that he does not want an heuristic ?
        if self.should_fit_kernel {
            self.kernel
                .heuristic_fit(&self.training_inputs, &self.training_outputs);
        }

        // builds a gp
        let mut gp = GaussianProcess::<KernelType, PriorType>::new(
            self.prior,
            self.kernel,
            self.noise,
            self.training_inputs,
            self.training_outputs,
        );

        // fit the model, if requested, on the training data
        gp.fit_parameters(
            self.should_fit_prior,
            self.should_fit_kernel,
            self.max_iter,
            self.convergence_fraction,
        );
        gp
    }
}
