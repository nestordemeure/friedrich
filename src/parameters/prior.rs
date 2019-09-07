//! Prior
//!
//! The value that is returned in the absence of further information.
//! This can be a constant but also a polynomial or any model.

use nalgebra::{DVector, DMatrix};

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
   fn prior(&self, input: &DVector<f64>) -> DVector<f64>;

   /// Optional, function that performs an automatic fit of the prior
   fn fit(&mut self, _training_inputs: &DMatrix<f64>, _training_outputs: &DMatrix<f64>) {}
}

//---------------------------------------------------------------------------------------
// CLASSICAL PRIOR

/// The Constant prior
#[derive(Clone, Debug)]
pub struct Constant
{
   /// Constant term.
   c: DVector<f64>
}

impl Constant
{
   /// Constructs a new constant prior
   pub fn new(c: DVector<f64>) -> Constant
   {
      Constant { c: c }
   }
}

impl Prior for Constant
{
   fn default(_input_dimension: usize, output_dimension: usize) -> Constant
   {
      Constant::new(DVector::zeros(output_dimension))
   }

   fn prior(&self, _input: &DVector<f64>) -> DVector<f64>
   {
      self.c.clone()
   }

   fn fit(&mut self, _training_inputs: &DMatrix<f64>, training_outputs: &DMatrix<f64>)
   {
      self.c = training_outputs.column_mean();
   }
}

//-----------------------------------------------

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

   fn prior(&self, _input: &DVector<f64>) -> DVector<f64>
   {
      DVector::zeros(self.output_dimension)
   }
}

//-----------------------------------------------

/// The Lenear prior
#[derive(Clone, Debug)]
pub struct Linear
{
   /// weights term.
   w: DMatrix<f64>,
   /// bias
   b: DVector<f64>
}

impl Linear
{
   /// Constructs a new linear prior
   pub fn new(w: DMatrix<f64>, b: DVector<f64>) -> Self
   {
      Linear { w, b }
   }
}

impl Prior for Linear
{
   fn default(input_dimension: usize, output_dimension: usize) -> Linear
   {
      let w = DMatrix::zeros(input_dimension, output_dimension);
      let b = DVector::zeros(output_dimension);
      Linear::new(w, b)
   }

   fn prior(&self, input: &DVector<f64>) -> DVector<f64>
   {
      input * &self.w + &self.b
   }

   /// performs a linear fit to set the value of the prior
   fn fit(&mut self, training_inputs: &DMatrix<f64>, training_outputs: &DMatrix<f64>)
   {
      let solution = training_inputs.clone()
                                    .insert_column(0, 1f64) // add constant term
                                    .lu()
                                    .solve(training_outputs) // solve linear system using LU decomposition
                                    .expect("Resolution of linear system failed");
      self.b = solution.row(0).transpose();
      self.w = solution.remove_row(0);
   }
}
//-----------------------------------------------

/// The arbitrary prior
pub struct Arbitrary
{
   /// arbitrary function
   f: Box<dyn Fn(&DVector<f64>) -> DVector<f64>>
}

impl Arbitrary
{
   /// Constructs a new arbitrary prior
   pub fn new(f: impl Fn(&DVector<f64>) -> DVector<f64> + 'static) -> Arbitrary
   {
      let f = Box::new(f);
      Arbitrary { f }
   }
}

impl Prior for Arbitrary
{
   fn default(_input_dimension: usize, output_dimension: usize) -> Arbitrary
   {
      let f = Box::new(move |_input: &DVector<f64>| DVector::zeros(output_dimension));
      Arbitrary { f }
   }

   fn prior(&self, input: &DVector<f64>) -> DVector<f64>
   {
      (self.f)(input)
   }
}
