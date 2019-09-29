use nalgebra::DMatrix;

/// converts some data into a valid input matrix
pub trait Input: Sized
{
   fn to_input(self) -> DMatrix<f64>;
}

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

