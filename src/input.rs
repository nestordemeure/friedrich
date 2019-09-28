use nalgebra::{DVector, DMatrix};

//-----------------------------------------------------------------------------
// INPUT

/// converts some data into a valid input matrix
pub trait Input
{
   fn to_input(self) -> DMatrix<f64>;
}

//-----------------------------------------------

// trivial implementation
impl Input for DMatrix<f64>
{
   fn to_input(self) -> DMatrix<f64>
   {
      self
   }
}

// implementation on single slice

// implementation on vector of slices

// implementation on slice of slices ?

//-----------------------------------------------------------------------------
// OUTPUT

/// converts some data into a valid output vector
pub trait Output
{
   fn to_output(self) -> DVector<f64>;
}

//-----------------------------------------------

// trivial implementation
impl Output for DVector<f64>
{
   fn to_output(self) -> DVector<f64>
   {
      self
   }
}

// implementation on Rust std::Vec

// implementation on slice

// implementation on single number
