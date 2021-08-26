//! # Kernels
//!
//! A kernel is a function that maps from two row vectors to a scalar which is used to express the similarity between the vectors.
//!
//! To learn more about the properties of the provided kernels, we recommand the [Usual_covariance_functions](https://en.wikipedia.org/wiki/Gaussian_process#Usual_covariance_functions) Wikipedia page and the [kernel-functions-for-machine-learning-applications](http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#kernel_functions) article.
//!
//! User-defined kernels should implement the Kernel trait.
//! To learn more about the implementation of kernels adapted to a particular problem, we recommend the chapter two (*Expressing Structure with Kernels*) and three (*Automatic Model Construction*) of the very good [Automatic Model Construction with Gaussian Processes](http://www.cs.toronto.edu/~duvenaud/thesis.pdf).
//!
//! This implementation is inspired by [rusty-machines'](https://github.com/AtheMathmo/rusty-machine/blob/master/src/learning/toolkit/kernel.rs).

use std::ops::{Add, Mul};
use nalgebra::{storage::Storage, U1, Dynamic};
use crate::algebra::{SRowVector, SVector, SMatrix};

//---------------------------------------------------------------------------------------
// TRAIT

/// The Kernel trait
///
/// If you want to provide a user-defined kernel, you should implement this trait.
pub trait Kernel: Default
{
   /// Numbers of parameters (such as bandwith and amplitude) of the kernel.
   const NB_PARAMETERS: usize;

   /// Can the kernel be rescaled (see the `rescale` function) ?
   /// This value is `false` by default.
   const IS_SCALABLE: bool = false; // TODO check whether more existing kernel can be made Is_SCALABLE

   /// Multiplies the amplitude of the kernel by the `scale` parameter such that a kernel `a*K(x,y)` becomes `scale*a*K(x,y)`
   ///
   /// When possible, do implement this function as it unlock a faster parameter fitting algorithm.
   ///
   /// *WARNING:* the code will panic if you set `IS_SCALABLE` to `true` without providing a user defined implementation of this function.
   fn rescale(&mut self, _scale: f64)
   {
      // TODO get rid of test and add ScalableKernel trait once specialization lands on stable
      if Self::IS_SCALABLE
      {
         unimplemented!("Please implement the `rescale` function if you set `IS_SCALABLE` to true.")
      }
      else
      {
         panic!("You tried to rescale a Kernel that is not Scalable!")
      }
   }
   /// Takes two equal length slices (row vector) and returns a scalar.
   ///
   /// NOTE: due to the optimization algorithm, this function might get illegal parameters (ie: negativ parameters),
   /// it is the duty of the function implementer to deal with them properly (ie : using an absolute value).
   fn kernel<S1: Storage<f64, U1, Dynamic>, S2: Storage<f64, U1, Dynamic>>(&self,
                                                                           x1: &SRowVector<S1>,
                                                                           x2: &SRowVector<S2>)
                                                                           -> f64;

   /// Takes two equal length slices (row vector) and returns a vector containing the value of the gradient for each parameter in an arbitrary order.
   ///
   /// NOTE: due to the optimization algorithm, this function might get illegal parameters (ie: negativ parameters),
   /// it is the duty of the function implementer to deal with them properly (ie: using the absolute value of the parameter and multiplying its gradient by its original sign).
   fn gradient<S1: Storage<f64, U1, Dynamic>, S2: Storage<f64, U1, Dynamic>>(&self,
                                                                             x1: &SRowVector<S1>,
                                                                             x2: &SRowVector<S2>)
                                                                             -> Vec<f64>;

   /// Returns a vector containing all the parameters of the kernel in the same order as the outputs of the `gradient` function
   fn get_parameters(&self) -> Vec<f64>;

   /// Sets all the parameters of the kernel by reading them from a slice where they are in the same order as the outputs of the `gradient` function
   fn set_parameters(&mut self, parameters: &[f64]);

   /// Optional, function that fits the kernel parameters on the training data using fast heuristics.
   /// This is used as a starting point for gradient descent.
   fn heuristic_fit<SM: Storage<f64, Dynamic, Dynamic>, SV: Storage<f64, Dynamic, U1>>(&mut self,
                                                                                       _training_inputs: &SMatrix<SM>,
                                                                                       _training_outputs: &SVector<SV>)
   {
   }
}

//---------------------------------------------------------------------------------------
// FIT

/// provides a rough estimate for the bandwith
///
/// use the mean distance between points as a baseline for the bandwith
fn fit_bandwith_mean<S: Storage<f64, Dynamic, Dynamic>>(training_inputs: &SMatrix<S>) -> f64
{
   // builds the sum of all distances between different samples
   let mut sum_distances = 0.;
   for (sample_index, sample) in training_inputs.row_iter().enumerate()
   {
      for sample2 in training_inputs.row_iter().skip(sample_index + 1)
      {
         let distance = (sample - sample2).norm();
         sum_distances += distance;
      }
   }

   // counts the number of distances that have been computed
   let nb_samples = training_inputs.nrows();
   let nb_distances = ((nb_samples * nb_samples - nb_samples) / 2) as f64;

   // mean distance
   sum_distances / nb_distances
}

/// outputs the variance of the outputs as a best guess of the amplitude
fn fit_amplitude_var<S: Storage<f64, Dynamic, U1>>(training_outputs: &SVector<S>) -> f64
{
   training_outputs.variance()
}

//---------------------------------------------------------------------------------------
// KERNEL COMBINAISON

/// The sum of two kernels
///
/// This struct should not be directly instantiated but instead is created when we add two kernels together.
///
/// Note that it will be more efficient to implement the final kernel manually yourself.
/// However this provides an easy mechanism to test different combinations.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "friedrich_serde", derive(serde::Deserialize, serde::Serialize))]
pub struct KernelSum<T, U>
   where T: Kernel,
         U: Kernel
{
   k1: T,
   k2: U
}

/// Computes the sum of the two associated kernels.
impl<T, U> Kernel for KernelSum<T, U>
   where T: Kernel,
         U: Kernel
{
   const NB_PARAMETERS: usize = T::NB_PARAMETERS + U::NB_PARAMETERS;
   const IS_SCALABLE: bool = T::IS_SCALABLE && U::IS_SCALABLE;

   fn kernel<S1: Storage<f64, U1, Dynamic>, S2: Storage<f64, U1, Dynamic>>(&self,
                                                                           x1: &SRowVector<S1>,
                                                                           x2: &SRowVector<S2>)
                                                                           -> f64
   {
      self.k1.kernel(x1, x2) + self.k2.kernel(x1, x2)
   }

   fn gradient<S1: Storage<f64, U1, Dynamic>, S2: Storage<f64, U1, Dynamic>>(&self,
                                                                             x1: &SRowVector<S1>,
                                                                             x2: &SRowVector<S2>)
                                                                             -> Vec<f64>
   {
      let mut g1 = self.k1.gradient(x1, x2);
      let mut g2 = self.k2.gradient(x1, x2);
      g1.append(&mut g2);
      g1
   }

   fn rescale(&mut self, scale: f64)
   {
      self.k1.rescale(scale);
      self.k2.rescale(scale);
   }

   fn get_parameters(&self) -> Vec<f64>
   {
      let mut p1 = self.k1.get_parameters();
      let mut p2 = self.k2.get_parameters();
      p1.append(&mut p2);
      p1
   }

   fn set_parameters(&mut self, parameters: &[f64])
   {
      self.k1.set_parameters(&parameters[..T::NB_PARAMETERS]);
      self.k2.set_parameters(&parameters[T::NB_PARAMETERS..]);
   }

   fn heuristic_fit<SM: Storage<f64, Dynamic, Dynamic>, SV: Storage<f64, Dynamic, U1>>(&mut self,
                                                                                       training_inputs: &SMatrix<SM>,
                                                                                       training_outputs: &SVector<SV>)
   {
      self.k1.heuristic_fit(training_inputs, training_outputs);
      self.k2.heuristic_fit(training_inputs, training_outputs);
   }
}

impl<T: Kernel, U: Kernel> Default for KernelSum<T, U>
{
   fn default() -> Self
   {
      let k1 = T::default();
      let k2 = U::default();
      KernelSum { k1, k2 }
   }
}

/// The pointwise product of two kernels
///
/// This struct should not be directly instantiated but instead is created when we multiply two kernels together.
///
/// Note that it will be more efficient to implement the final kernel manually yourself.
/// However this provides an easy mechanism to test different combinations.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "friedrich_serde", derive(serde::Deserialize, serde::Serialize))]
pub struct KernelProd<T, U>
   where T: Kernel,
         U: Kernel
{
   k1: T,
   k2: U
}

/// Computes the product of the two associated kernels.
impl<T, U> Kernel for KernelProd<T, U>
   where T: Kernel,
         U: Kernel
{
   const NB_PARAMETERS: usize = T::NB_PARAMETERS + U::NB_PARAMETERS;
   const IS_SCALABLE: bool = T::IS_SCALABLE || U::IS_SCALABLE;

   fn kernel<S1: Storage<f64, U1, Dynamic>, S2: Storage<f64, U1, Dynamic>>(&self,
                                                                           x1: &SRowVector<S1>,
                                                                           x2: &SRowVector<S2>)
                                                                           -> f64
   {
      self.k1.kernel(x1, x2) * self.k2.kernel(x1, x2)
   }

   fn gradient<S1: Storage<f64, U1, Dynamic>, S2: Storage<f64, U1, Dynamic>>(&self,
                                                                             x1: &SRowVector<S1>,
                                                                             x2: &SRowVector<S2>)
                                                                             -> Vec<f64>
   {
      let k1 = self.k1.kernel(x1, x2);
      let k2 = self.k2.kernel(x1, x2);
      let g1 = self.k1.gradient(x1, x2);
      let g2 = self.k2.gradient(x1, x2);
      g1.iter().map(|g1| g1 * k2).chain(g2.iter().map(|g2| g2 * k1)).collect()
   }

   fn rescale(&mut self, scale: f64)
   {
      if T::IS_SCALABLE
      {
         self.k1.rescale(scale);
      }
      else
      {
         self.k2.rescale(scale);
      }
   }

   fn get_parameters(&self) -> Vec<f64>
   {
      let mut p1 = self.k1.get_parameters();
      let mut p2 = self.k2.get_parameters();
      p1.append(&mut p2);
      p1
   }

   fn set_parameters(&mut self, parameters: &[f64])
   {
      self.k1.set_parameters(&parameters[..T::NB_PARAMETERS]);
      self.k2.set_parameters(&parameters[T::NB_PARAMETERS..]);
   }

   fn heuristic_fit<SM: Storage<f64, Dynamic, Dynamic>, SV: Storage<f64, Dynamic, U1>>(&mut self,
                                                                                       training_inputs: &SMatrix<SM>,
                                                                                       training_outputs: &SVector<SV>)
   {
      self.k1.heuristic_fit(training_inputs, training_outputs);
      self.k2.heuristic_fit(training_inputs, training_outputs);
   }
}

impl<T: Kernel, U: Kernel> Default for KernelProd<T, U>
{
   fn default() -> Self
   {
      let k1 = T::default();
      let k2 = U::default();
      KernelProd { k1, k2 }
   }
}

/// A wrapper tuple struct used for kernel arithmetic
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "friedrich_serde", derive(serde::Deserialize, serde::Serialize))]
pub struct KernelArith<K: Kernel>(pub K);

impl<T: Kernel, U: Kernel> Add<KernelArith<T>> for KernelArith<U>
{
   type Output = KernelSum<U, T>;

   fn add(self, ker: KernelArith<T>) -> KernelSum<U, T>
   {
      KernelSum { k1: self.0, k2: ker.0 }
   }
}

impl<T: Kernel, U: Kernel> Mul<KernelArith<T>> for KernelArith<U>
{
   type Output = KernelProd<U, T>;

   fn mul(self, ker: KernelArith<T>) -> KernelProd<U, T>
   {
      KernelProd { k1: self.0, k2: ker.0 }
   }
}

//---------------------------------------------------------------------------------------
// CLASSICAL KERNELS

/// The Linear Kernel
///
/// k(x,y) = x^Ty + c
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "friedrich_serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Linear
{
   /// Constant term added to inner product.
   pub c: f64
}

impl Linear
{
   /// Constructs a new Linear Kernel.
   pub fn new(c: f64) -> Linear
   {
      Linear { c: c }
   }
}

/// Constructs the default Linear Kernel
///
/// The defaults are:
/// - c = 0
impl Default for Linear
{
   fn default() -> Linear
   {
      Linear { c: 0f64 }
   }
}

impl Kernel for Linear
{
   const NB_PARAMETERS: usize = 1;

   fn kernel<S1: Storage<f64, U1, Dynamic>, S2: Storage<f64, U1, Dynamic>>(&self,
                                                                           x1: &SRowVector<S1>,
                                                                           x2: &SRowVector<S2>)
                                                                           -> f64
   {
      x1.dot(&x2) + self.c
   }

   fn gradient<S1: Storage<f64, U1, Dynamic>, S2: Storage<f64, U1, Dynamic>>(&self,
                                                                             _x1: &SRowVector<S1>,
                                                                             _x2: &SRowVector<S2>)
                                                                             -> Vec<f64>
   {
      let grad_c = 1.;
      vec![grad_c]
   }

   fn get_parameters(&self) -> Vec<f64>
   {
      vec![self.c]
   }

   fn set_parameters(&mut self, parameters: &[f64])
   {
      self.c = parameters[0];
   }
}

//-----------------------------------------------

/// The Polynomial Kernel
///
/// k(x,y) = (αx^Ty + c)^d
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "friedrich_serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Polynomial
{
   /// Scaling of the inner product.
   pub alpha: f64,
   /// Constant added to inner product.
   pub c: f64,
   /// The power to raise the sum to.
   pub d: f64
}

impl Polynomial
{
   /// Constructs a new Polynomial Kernel.
   pub fn new(alpha: f64, c: f64, d: f64) -> Polynomial
   {
      Polynomial { alpha: alpha, c: c, d: d }
   }
}

/// Construct a new polynomial kernel.
///
/// The defaults are:
/// - alpha = 1
/// - c = 0
/// - d = 1
impl Default for Polynomial
{
   fn default() -> Polynomial
   {
      Polynomial { alpha: 1f64, c: 0f64, d: 1f64 }
   }
}

impl Kernel for Polynomial
{
   const NB_PARAMETERS: usize = 3;

   fn kernel<S1: Storage<f64, U1, Dynamic>, S2: Storage<f64, U1, Dynamic>>(&self,
                                                                           x1: &SRowVector<S1>,
                                                                           x2: &SRowVector<S2>)
                                                                           -> f64
   {
      (self.alpha * x1.dot(&x2) + self.c).powf(self.d)
   }

   fn gradient<S1: Storage<f64, U1, Dynamic>, S2: Storage<f64, U1, Dynamic>>(&self,
                                                                             x1: &SRowVector<S1>,
                                                                             x2: &SRowVector<S2>)
                                                                             -> Vec<f64>
   {
      let x = x1.dot(&x2);
      let inner_term = self.alpha * x + self.c;

      let grad_c = self.d * inner_term.powf(self.d - 1.);
      let grad_alpha = x * grad_c;
      let grad_d = inner_term.ln() * inner_term.powf(self.d);

      vec![grad_alpha, grad_c, grad_d]
   }

   fn get_parameters(&self) -> Vec<f64>
   {
      vec![self.alpha, self.c, self.d]
   }

   fn set_parameters(&mut self, parameters: &[f64])
   {
      self.alpha = parameters[0];
      self.c = parameters[1];
      self.d = parameters[2];
   }
}

//-----------------------------------------------

/// Gaussian kernel
///
/// Equivalent to a squared exponential kernel.
///
/// k(x,y) = A exp(-||x-y||² / 2l²)
///
/// Where A is the amplitude and l the length scale.
pub type Gaussian = SquaredExp;

/// Squared exponential kernel
///
/// Equivalent to a gaussian kernel.
///
/// k(x,y) = A exp(-||x-y||² / 2l²)
///
/// Where A is the amplitude and l the length scale.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "friedrich_serde", derive(serde::Deserialize, serde::Serialize))]
pub struct SquaredExp
{
   /// The length scale of the kernel.
   pub ls: f64,
   /// The amplitude of the kernel.
   pub ampl: f64
}

impl SquaredExp
{
   /// Construct a new squared exponential kernel (gaussian).
   pub fn new(ls: f64, ampl: f64) -> SquaredExp
   {
      SquaredExp { ls: ls, ampl: ampl }
   }
}

/// Constructs the default Squared Exp kernel.
///
/// The defaults are:
/// - ls = 1
/// - ampl = 1
impl Default for SquaredExp
{
   fn default() -> SquaredExp
   {
      SquaredExp { ls: 1f64, ampl: 1f64 }
   }
}

impl Kernel for SquaredExp
{
   const NB_PARAMETERS: usize = 2;
   const IS_SCALABLE: bool = true;

   /// The squared exponential kernel function.
   fn kernel<S1: Storage<f64, U1, Dynamic>, S2: Storage<f64, U1, Dynamic>>(&self,
                                                                           x1: &SRowVector<S1>,
                                                                           x2: &SRowVector<S2>)
                                                                           -> f64
   {
      // sanitize parameters
      let ampl = self.ampl.abs();
      // computes kernel
      let distance_squared = (x1 - x2).norm_squared();
      let x = -distance_squared / (2f64 * self.ls * self.ls);
      ampl * x.exp()
   }

   fn gradient<S1: Storage<f64, U1, Dynamic>, S2: Storage<f64, U1, Dynamic>>(&self,
                                                                             x1: &SRowVector<S1>,
                                                                             x2: &SRowVector<S2>)
                                                                             -> Vec<f64>
   {
      // sanitize parameters
      let ampl = self.ampl.abs();
      // compute gradients
      let distance_squared = (x1 - x2).norm_squared();
      let exponential = (-distance_squared / (2f64 * self.ls * self.ls)).exp();
      let grad_ls = (distance_squared * ampl * exponential) / self.ls.powi(3);
      let grad_ampl = self.ampl.signum() * exponential;
      vec![grad_ls, grad_ampl]
   }

   fn rescale(&mut self, scale: f64)
   {
      self.ampl *= scale;
   }

   fn get_parameters(&self) -> Vec<f64>
   {
      vec![self.ls, self.ampl]
   }

   fn set_parameters(&mut self, parameters: &[f64])
   {
      self.ls = parameters[0];
      self.ampl = parameters[1];
   }

   fn heuristic_fit<SM: Storage<f64, Dynamic, Dynamic>, SV: Storage<f64, Dynamic, U1>>(&mut self,
                                                                                       training_inputs: &SMatrix<SM>,
                                                                                       training_outputs: &SVector<SV>)
   {
      self.ls = fit_bandwith_mean(training_inputs);
      self.ampl = fit_amplitude_var(training_outputs);
   }
}

//-----------------------------------------------

/// The Exponential Kernel
///
/// k(x,y) = A exp(-||x-y|| / 2l²)
///
/// Where A is the amplitude and l is the length scale.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "friedrich_serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Exponential
{
   /// The length scale of the kernel.
   pub ls: f64,
   /// The amplitude of the kernel.
   pub ampl: f64
}

impl Exponential
{
   /// Construct a new squared exponential kernel.
   pub fn new(ls: f64, ampl: f64) -> Exponential
   {
      Exponential { ls: ls, ampl: ampl }
   }
}

/// Constructs the default Exponential kernel.
///
/// The defaults are:
/// - ls = 1
/// - amplitude = 1
impl Default for Exponential
{
   fn default() -> Exponential
   {
      Exponential { ls: 1f64, ampl: 1f64 }
   }
}

impl Kernel for Exponential
{
   const NB_PARAMETERS: usize = 2;
   const IS_SCALABLE: bool = true;

   /// The squared exponential kernel function.
   fn kernel<S1: Storage<f64, U1, Dynamic>, S2: Storage<f64, U1, Dynamic>>(&self,
                                                                           x1: &SRowVector<S1>,
                                                                           x2: &SRowVector<S2>)
                                                                           -> f64
   {
      // sanitize parameters
      let ampl = self.ampl.abs();
      // compute kernel
      let distance = (x1 - x2).norm();
      let x = -distance / (2f64 * self.ls * self.ls);
      ampl * x.exp()
   }

   fn gradient<S1: Storage<f64, U1, Dynamic>, S2: Storage<f64, U1, Dynamic>>(&self,
                                                                             x1: &SRowVector<S1>,
                                                                             x2: &SRowVector<S2>)
                                                                             -> Vec<f64>
   {
      // sanitize parameters
      let ampl = self.ampl.abs();
      // compute gradients
      let distance = (x1 - x2).norm();
      let exponential = (-distance / (2f64 * self.ls * self.ls)).exp();
      let grad_ls = (distance * ampl * exponential) / self.ls.powi(3);
      let grad_ampl = self.ampl.signum() * exponential;
      vec![grad_ls, grad_ampl]
   }

   fn rescale(&mut self, scale: f64)
   {
      self.ampl *= scale;
   }

   fn get_parameters(&self) -> Vec<f64>
   {
      vec![self.ls, self.ampl]
   }

   fn set_parameters(&mut self, parameters: &[f64])
   {
      self.ls = parameters[0];
      self.ampl = parameters[1];
   }

   fn heuristic_fit<SM: Storage<f64, Dynamic, Dynamic>, SV: Storage<f64, Dynamic, U1>>(&mut self,
                                                                                       training_inputs: &SMatrix<SM>,
                                                                                       training_outputs: &SVector<SV>)
   {
      self.ls = fit_bandwith_mean(training_inputs);
      self.ampl = fit_amplitude_var(training_outputs);
   }
}

//-----------------------------------------------

/// The Matèrn1 kernel which is 1 differentiable and correspond to a classical Matèrn kernel with nu=3/2
///
/// k(x,y) = A (1 + ||x-y||sqrt(3)/l) exp(-||x-y||sqrt(3)/l)
///
/// Where A is the amplitude and l is the length scale.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "friedrich_serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Matern1
{
   /// The length scale of the kernel.
   pub ls: f64,
   /// The amplitude of the kernel.
   pub ampl: f64
}

impl Matern1
{
   /// Construct a new matèrn1 kernel.
   pub fn new(ls: f64, ampl: f64) -> Matern1
   {
      Matern1 { ls: ls, ampl: ampl }
   }
}

/// Constructs the default Matern1 kernel.
///
/// The defaults are:
/// - ls = 1
/// - amplitude = 1
impl Default for Matern1
{
   fn default() -> Matern1
   {
      Matern1 { ls: 1f64, ampl: 1f64 }
   }
}

impl Kernel for Matern1
{
   const NB_PARAMETERS: usize = 2;
   const IS_SCALABLE: bool = true;

   /// The matèrn1 kernel function.
   fn kernel<S1: Storage<f64, U1, Dynamic>, S2: Storage<f64, U1, Dynamic>>(&self,
                                                                           x1: &SRowVector<S1>,
                                                                           x2: &SRowVector<S2>)
                                                                           -> f64
   {
      // sanitize parameters
      let ampl = self.ampl.abs();
      let l = self.ls.abs();
      // compute kernel
      let distance = (x1 - x2).norm();
      let x = (3f64).sqrt() * distance / l;
      ampl * (1f64 + x) * (-x).exp()
   }

   fn gradient<S1: Storage<f64, U1, Dynamic>, S2: Storage<f64, U1, Dynamic>>(&self,
                                                                             x1: &SRowVector<S1>,
                                                                             x2: &SRowVector<S2>)
                                                                             -> Vec<f64>
   {
      // sanitize parameters
      let ampl = self.ampl.abs();
      let l = self.ls.abs();
      // compute gradient
      let distance = (x1 - x2).norm();
      let x = 3f64.sqrt() * distance / l;
      let grad_ls = (3. * ampl * distance.powi(2) * (-x).exp()) / (self.ls.powi(3));
      let grad_ampl = self.ampl.signum() * (1. + x) * (-x).exp();
      vec![grad_ls, grad_ampl]
   }

   fn rescale(&mut self, scale: f64)
   {
      self.ampl *= scale;
   }

   fn get_parameters(&self) -> Vec<f64>
   {
      vec![self.ls, self.ampl]
   }

   fn set_parameters(&mut self, parameters: &[f64])
   {
      self.ls = parameters[0];
      self.ampl = parameters[1];
   }

   fn heuristic_fit<SM: Storage<f64, Dynamic, Dynamic>, SV: Storage<f64, Dynamic, U1>>(&mut self,
                                                                                       training_inputs: &SMatrix<SM>,
                                                                                       training_outputs: &SVector<SV>)
   {
      self.ls = fit_bandwith_mean(training_inputs);
      self.ampl = fit_amplitude_var(training_outputs);
   }
}

//-----------------------------------------------

/// The Matèrn2 kernel which is 2 differentiable and correspond to a classical Matèrn kernel with nu=5/2
///
/// k(x,y) = A (1 + ||x-y||sqrt(5)/l + ||x-y||²5/3l²) exp(-||x-y||sqrt(5)/l)
///
/// Where A is the amplitude and l is the length scale.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "friedrich_serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Matern2
{
   /// The length scale of the kernel.
   pub ls: f64,
   /// The amplitude of the kernel.
   pub ampl: f64
}

impl Matern2
{
   /// Construct a new matèrn2 kernel.
   pub fn new(ls: f64, ampl: f64) -> Matern2
   {
      Matern2 { ls: ls, ampl: ampl }
   }
}

/// Constructs the default Matern2 kernel.
///
/// The defaults are:
/// - ls = 1
/// - amplitude = 1
impl Default for Matern2
{
   fn default() -> Matern2
   {
      Matern2 { ls: 1f64, ampl: 1f64 }
   }
}

impl Kernel for Matern2
{
   const NB_PARAMETERS: usize = 2;
   const IS_SCALABLE: bool = true;

   /// The matèrn2 kernel function.
   fn kernel<S1: Storage<f64, U1, Dynamic>, S2: Storage<f64, U1, Dynamic>>(&self,
                                                                           x1: &SRowVector<S1>,
                                                                           x2: &SRowVector<S2>)
                                                                           -> f64
   {
      // sanitize parameters
      let ampl = self.ampl.abs();
      let l = self.ls.abs();
      // compute kernel
      let distance = (x1 - x2).norm();
      let x = (5f64).sqrt() * distance / l;
      ampl * (1f64 + x + (5f64 * distance * distance) / (3f64 * l * l)) * (-x).exp()
   }

   fn gradient<S1: Storage<f64, U1, Dynamic>, S2: Storage<f64, U1, Dynamic>>(&self,
                                                                             x1: &SRowVector<S1>,
                                                                             x2: &SRowVector<S2>)
                                                                             -> Vec<f64>
   {
      // sanitize parameters
      let ampl = self.ampl.abs();
      let l = self.ls.abs();
      // compute gradient
      let distance = (x1 - x2).norm();
      let x = (5f64).sqrt() * distance / self.ls;
      let grad_ls = self.ls.signum()
                    * ampl
                    * ((2. * l / 3. + 1.) + distance * 5f64.sqrt() * ((l.powi(2) / 3. + l + 1.) / l.powi(2)))
                    * (-x).exp();
      let grad_ampl =
         self.ampl.signum() * (1f64 + x + (5f64 * distance * distance) / (3f64 * l * l)) * (-x).exp();
      vec![grad_ls, grad_ampl]
   }

   fn rescale(&mut self, scale: f64)
   {
      self.ampl *= scale;
   }

   fn get_parameters(&self) -> Vec<f64>
   {
      vec![self.ls, self.ampl]
   }

   fn set_parameters(&mut self, parameters: &[f64])
   {
      self.ls = parameters[0];
      self.ampl = parameters[1];
   }

   fn heuristic_fit<SM: Storage<f64, Dynamic, Dynamic>, SV: Storage<f64, Dynamic, U1>>(&mut self,
                                                                                       training_inputs: &SMatrix<SM>,
                                                                                       training_outputs: &SVector<SV>)
   {
      self.ls = fit_bandwith_mean(training_inputs);
      self.ampl = fit_amplitude_var(training_outputs);
   }
}

//-----------------------------------------------

/// The Hyperbolic Tangent Kernel.
///
/// ker(x,y) = tanh(αx^Ty + c)
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "friedrich_serde", derive(serde::Deserialize, serde::Serialize))]
pub struct HyperTan
{
   /// The scaling of the inner product.
   pub alpha: f64,
   /// The constant to add to the inner product.
   pub c: f64
}

impl HyperTan
{
   /// Constructs a new Hyperbolic Tangent Kernel.
   pub fn new(alpha: f64, c: f64) -> HyperTan
   {
      HyperTan { alpha: alpha, c: c }
   }
}

/// Constructs a default Hyperbolic Tangent Kernel.
///
/// The defaults are:
/// - alpha = 1
/// - c = 0
impl Default for HyperTan
{
   fn default() -> HyperTan
   {
      HyperTan { alpha: 1f64, c: 0f64 }
   }
}

impl Kernel for HyperTan
{
   const NB_PARAMETERS: usize = 2;

   fn kernel<S1: Storage<f64, U1, Dynamic>, S2: Storage<f64, U1, Dynamic>>(&self,
                                                                           x1: &SRowVector<S1>,
                                                                           x2: &SRowVector<S2>)
                                                                           -> f64
   {
      (self.alpha * x1.dot(&x2) + self.c).tanh()
   }

   fn gradient<S1: Storage<f64, U1, Dynamic>, S2: Storage<f64, U1, Dynamic>>(&self,
                                                                             x1: &SRowVector<S1>,
                                                                             x2: &SRowVector<S2>)
                                                                             -> Vec<f64>
   {
      let x = x1.dot(&x2);
      let grad_c = 1. / (self.alpha * x + self.c).cosh().powi(2);
      let grad_alpha = x * grad_c;

      vec![grad_alpha, grad_c]
   }

   fn get_parameters(&self) -> Vec<f64>
   {
      vec![self.alpha, self.c]
   }

   fn set_parameters(&mut self, parameters: &[f64])
   {
      self.alpha = parameters[0];
      self.c = parameters[1];
   }
}

//-----------------------------------------------

/// The Multiquadric Kernel.
///
/// k(x,y) = sqrt(||x-y||² + c²)
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "friedrich_serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Multiquadric
{
   /// Constant added to square of difference.
   pub c: f64
}

impl Multiquadric
{
   /// Constructs a new Multiquadric Kernel.
   pub fn new(c: f64) -> Multiquadric
   {
      Multiquadric { c: c }
   }
}

/// Constructs a default Multiquadric Kernel.
///
/// The defaults are:
/// - c = 0
impl Default for Multiquadric
{
   fn default() -> Multiquadric
   {
      Multiquadric { c: 0f64 }
   }
}

impl Kernel for Multiquadric
{
   const NB_PARAMETERS: usize = 2;

   fn kernel<S1: Storage<f64, U1, Dynamic>, S2: Storage<f64, U1, Dynamic>>(&self,
                                                                           x1: &SRowVector<S1>,
                                                                           x2: &SRowVector<S2>)
                                                                           -> f64
   {
      (x1 - x2).norm_squared().hypot(self.c)
   }

   fn gradient<S1: Storage<f64, U1, Dynamic>, S2: Storage<f64, U1, Dynamic>>(&self,
                                                                             x1: &SRowVector<S1>,
                                                                             x2: &SRowVector<S2>)
                                                                             -> Vec<f64>
   {
      let grad_c = self.c / (x1 - x2).norm().hypot(self.c);
      vec![grad_c]
   }

   fn get_parameters(&self) -> Vec<f64>
   {
      vec![self.c]
   }

   fn set_parameters(&mut self, parameters: &[f64])
   {
      self.c = parameters[1];
   }
}

//-----------------------------------------------

/// The Rational Quadratic Kernel.
///
/// k(x,y) = (1 + ||x-y||² / (2αl²))^-α
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "friedrich_serde", derive(serde::Deserialize, serde::Serialize))]
pub struct RationalQuadratic
{
   /// Controls inverse power and difference scale.
   pub alpha: f64,
   /// Length scale controls scale of difference.
   pub ls: f64
}

impl RationalQuadratic
{
   /// Constructs a new Rational Quadratic Kernel.
   pub fn new(alpha: f64, ls: f64) -> RationalQuadratic
   {
      RationalQuadratic { alpha: alpha, ls: ls }
   }
}

/// The default Rational Quadratic Kernel.
///
/// The defaults are:
/// - alpha = 1
/// - ls = 1
impl Default for RationalQuadratic
{
   fn default() -> RationalQuadratic
   {
      RationalQuadratic { alpha: 1f64, ls: 1f64 }
   }
}

impl Kernel for RationalQuadratic
{
   const NB_PARAMETERS: usize = 2;

   fn kernel<S1: Storage<f64, U1, Dynamic>, S2: Storage<f64, U1, Dynamic>>(&self,
                                                                           x1: &SRowVector<S1>,
                                                                           x2: &SRowVector<S2>)
                                                                           -> f64
   {
      let distance_squared = (x1 - x2).norm_squared();
      (1f64 + distance_squared / (2f64 * self.alpha * self.ls * self.ls)).powf(-self.alpha)
   }

   fn gradient<S1: Storage<f64, U1, Dynamic>, S2: Storage<f64, U1, Dynamic>>(&self,
                                                                             x1: &SRowVector<S1>,
                                                                             x2: &SRowVector<S2>)
                                                                             -> Vec<f64>
   {
      // sanitize parameters
      let l = self.ls.abs();
      // compute gradient
      let distance_squared = (x1 - x2).norm_squared();
      let grad_alpha =
         ((distance_squared + 2. * l.powi(2) * self.alpha) / (l.powi(2) * self.alpha)).powf(-self.alpha)
         * (2f64.powf(self.alpha)
            * (1. - ((distance_squared + 2. * l.powi(2) * self.alpha) / (2. * l.powi(2) * self.alpha)).ln())
            - (l.powi(2) * 2f64.powf(self.alpha + 1.) * self.alpha)
              / (distance_squared + 2. * l.powi(2) * self.alpha));
      let grad_ls = distance_squared
                    * (distance_squared / (2. * self.alpha * l * l) + 1.).powf(-self.alpha - 1.)
                    / self.ls.powi(3);
      vec![grad_alpha, grad_ls]
   }

   fn get_parameters(&self) -> Vec<f64>
   {
      vec![self.alpha, self.ls]
   }

   fn set_parameters(&mut self, parameters: &[f64])
   {
      self.alpha = parameters[0];
      self.ls = parameters[1];
   }
}
