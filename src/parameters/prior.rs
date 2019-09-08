//! Prior
//!
//! The value that is returned in the absence of further information.
//! This can be a constant but also a polynomial or any model.

use nalgebra::{DMatrix, RowDVector};
use crate::matrix::RowVectorSlice;

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
   fn prior(&self, input: RowVectorSlice) -> RowDVector<f64>;

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
   fn default(input_dimension: usize, output_dimension: usize) -> Self
   {
      Zero { output_dimension }
   }

   fn prior(&self, _input: RowVectorSlice) -> RowDVector<f64>
   {
      RowDVector::zeros(self.output_dimension)
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

   fn prior(&self, _input: RowVectorSlice) -> RowDVector<f64>
   {
      self.c.clone()
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
   /// weights term, the first row correspond to the bias.
   w: DMatrix<f64>
}

impl Linear
{
   /// Constructs a new linear prior
   /// TODO document the fact that the first row is the bias
   pub fn new(w: DMatrix<f64>) -> Self
   {
      Linear { w }
   }
}

impl Prior for Linear
{
   fn default(input_dimension: usize, output_dimension: usize) -> Linear
   {
      let w = DMatrix::zeros(input_dimension, output_dimension);
      Linear::new(w)
   }

   fn prior(&self, input: RowVectorSlice) -> RowDVector<f64>
   {
      //input * &self.w + &self.b
      input * self.w.rows(1, self.w.nrows() - 1) + self.w.row(0)
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
//-----------------------------------------------

/// The arbitrary prior
pub struct Arbitrary
{
   /// arbitrary function
   f: Box<dyn Fn(RowVectorSlice) -> RowDVector<f64>>
}

impl Arbitrary
{
   /// Constructs a new arbitrary prior
   pub fn new(f: impl Fn(RowVectorSlice) -> RowDVector<f64> + 'static) -> Arbitrary
   {
      let f = Box::new(f);
      Arbitrary { f }
   }
}

impl Prior for Arbitrary
{
   fn default(_input_dimension: usize, output_dimension: usize) -> Arbitrary
   {
      let f = Box::new(move |_input: RowVectorSlice| RowDVector::zeros(output_dimension));
      Arbitrary { f }
   }

   fn prior(&self, input: RowVectorSlice) -> RowDVector<f64>
   {
      (self.f)(input)
   }
}
