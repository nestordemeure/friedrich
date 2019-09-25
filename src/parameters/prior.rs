//! Prior
//!
//! The value that is returned in the absence of further information.
//! This can be a constant but also a polynomial or any model.

use nalgebra::{DMatrix, DVector};

//---------------------------------------------------------------------------------------
// TRAIT

/// The Prior trait
///
/// Requires a function mapping an input to an output
pub trait Prior
{
   /// Default value for the prior
   fn default(input_dimension: usize) -> Self;

   /// Takes and input and return an output.
   fn prior(&self, input: &DMatrix<f64>) -> DVector<f64>;

   /// Optional, function that performs an automatic fit of the prior
   fn fit(&mut self, _training_inputs: &DMatrix<f64>, _training_outputs: &DVector<f64>) {}
}

//---------------------------------------------------------------------------------------
// CLASSICAL PRIOR

/// The Zero prior
#[derive(Clone, Copy, Debug)]
pub struct Zero {}

impl Prior for Zero
{
   fn default(_input_dimension: usize) -> Self
   {
      Zero {}
   }

   fn prior(&self, input: &DMatrix<f64>) -> DVector<f64>
   {
      DVector::zeros(input.nrows())
   }
}

//-----------------------------------------------

/// The Constant prior
#[derive(Clone, Debug)]
pub struct Constant
{
   c: f64
}

impl Constant
{
   /// Constructs a new constant prior
   pub fn new(c: f64) -> Constant
   {
      Constant { c: c }
   }
}

impl Prior for Constant
{
   fn default(_input_dimension: usize) -> Constant
   {
      Constant::new(0f64)
   }

   fn prior(&self, input: &DMatrix<f64>) -> DVector<f64>
   {
      DVector::from_element(input.nrows(), self.c)
   }

   /// the prior is fitted on the mean of the training outputs
   fn fit(&mut self, _training_inputs: &DMatrix<f64>, training_outputs: &DVector<f64>)
   {
      self.c = training_outputs.mean();
   }
}

//-----------------------------------------------

/// The Linear prior
#[derive(Clone, Debug)]
pub struct Linear
{
   weights: DVector<f64>,
   intercept: f64
}

impl Linear
{
   /// Constructs a new linear prior
   /// te first row of w is the bias such that `prior = [1|input] * w`
   pub fn new(weights: DVector<f64>, intercept:f64) -> Self
   {
      Linear { weights, intercept }
   }
}

impl Prior for Linear
{
   fn default(input_dimension: usize) -> Linear
   {
      Linear { weights: DVector::zeros(input_dimension), intercept:0f64 }
   }

   fn prior(&self, input: &DMatrix<f64>) -> DVector<f64>
   {
      let mut result = input * &self.weights;
      result.add_scalar_mut(self.intercept);
      result
   }

   /// performs a linear fit to set the value of the prior
   fn fit(&mut self, training_inputs: &DMatrix<f64>, training_outputs: &DVector<f64>)
   {
      // solve linear system using LU decomposition
      let weights = training_inputs.clone()
                              .insert_column(0, 1f64) // add constant term for non-zero intercept
                              .lu()
                              .solve(training_outputs)
                              .expect("Resolution of linear system failed");
      // extracts weights and intercept
      self.intercept = weights[0];
      self.weights = weights.remove_row(0);
   }
}
