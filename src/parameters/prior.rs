//! Prior
//!
//! The value that is returned in the absence of further information.
//! This can be a constant but also a polynomial or any model.

use nalgebra::{DMatrix, RowDVector};
use crate::matrix;

//---------------------------------------------------------------------------------------
// TRAIT

/// The Prior trait
///
/// Requires a function mapping an input to an output
pub trait Prior
{
   /// Default value for the prior
   fn default(input_dimension: usize, output_dimenssion: usize) -> Self;

   /// Takes and input and return an output.
   fn prior(&self, input: &DMatrix<f64>) -> DMatrix<f64>;

   /// Optional, function that performs an automatic fit of the prior
   fn fit(&mut self, _training_inputs: &DMatrix<f64>, _training_outputs: &DMatrix<f64>) {}
}

//---------------------------------------------------------------------------------------
// CLASSICAL PRIOR

/// The Zero prior
#[derive(Clone, Copy, Debug)]
pub struct Zero
{
   output_dimension: usize
}

impl Prior for Zero
{
   fn default(_input_dimension: usize, output_dimension: usize) -> Self
   {
      Zero { output_dimension }
   }

   fn prior(&self, input: &DMatrix<f64>) -> DMatrix<f64>
   {
      DMatrix::zeros(input.nrows(), self.output_dimension)
   }
}

//-----------------------------------------------

/// The Constant prior
#[derive(Clone, Debug)]
pub struct Constant
{
   /// Constant term.
   c: RowDVector<f64>
}

impl Constant
{
   /// Constructs a new constant prior
   pub fn new(c: RowDVector<f64>) -> Constant
   {
      Constant { c: c }
   }
}

impl Prior for Constant
{
   fn default(_input_dimension: usize, output_dimension: usize) -> Constant
   {
      Constant::new(RowDVector::zeros(output_dimension))
   }

   fn prior(&self, input: &DMatrix<f64>) -> DMatrix<f64>
   {
      // TODO is there a faster way to build matrix from given row ?
      matrix::one(input.nrows(), 1) * &self.c
   }

   fn fit(&mut self, _training_inputs: &DMatrix<f64>, training_outputs: &DMatrix<f64>)
   {
      self.c = training_outputs.column_mean().transpose();
   }
}

//-----------------------------------------------

/// The Lenear prior
#[derive(Clone, Debug)]
pub struct Linear
{
   /// weight matrix : `prior = [1|input] * w`
   w: DMatrix<f64>
}

impl Linear
{
   /// Constructs a new linear prior
   /// te first row of w is the bias such that `prior = [1|input] * w`
   pub fn new(w: DMatrix<f64>) -> Self
   {
      Linear { w }
   }
}

impl Prior for Linear
{
   fn default(input_dimension: usize, output_dimension: usize) -> Linear
   {
      Linear { w: DMatrix::zeros(input_dimension + 1, output_dimension) }
   }

   fn prior(&self, input: &DMatrix<f64>) -> DMatrix<f64>
   {
      // TODO is there a faster way to add bias
      input * self.w.rows(1, self.w.nrows() - 1)
      + matrix::one(input.nrows(), 1) * self.w.row(0)
   }

   /// performs a linear fit to set the value of the prior
   fn fit(&mut self, training_inputs: &DMatrix<f64>, training_outputs: &DMatrix<f64>)
   {
      self.w = training_inputs.clone()
                              .insert_column(0, 1f64) // add constant term
                              .lu()
                              .solve(training_outputs) // solve linear system using LU decomposition
                              .expect("Resolution of linear system failed");
   }
}
