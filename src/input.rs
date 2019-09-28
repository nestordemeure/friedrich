use nalgebra::{DVector, DMatrix};

//-----------------------------------------------------------------------------
// INPUT

/// converts some data into a valid input matrix
pub trait Input: Sized
{
   fn to_input(&self) -> DMatrix<f64>;

   fn into_input(self) -> DMatrix<f64>
   {
      self.to_input()
   }
}

//-----------------------------------------------

// trivial implementation
impl Input for DMatrix<f64>
{
   fn to_input(&self) -> DMatrix<f64>
   {
      self.clone()
   }
}

// implementation on single slice

// implementation on vector of slices

// implementation on slice of slices ?

//-----------------------------------------------------------------------------
// OUTPUT

/// converts some data into a valid output vector
pub trait Output: Sized
{
   fn to_output(&self) -> DVector<f64>;

   fn into_output(self) -> DVector<f64>
   {
      self.to_output()
   }
}

//-----------------------------------------------

// trivial implementation
impl Output for DVector<f64>
{
   fn to_output(&self) -> DVector<f64>
   {
      self.clone()
   }
}

// implementation on Rust std::Vec

// implementation on slice

// implementation on single number
