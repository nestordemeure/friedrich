use nalgebra::DMatrix;

/// converts some data into a valid input matrix
pub trait AsMatrix: Sized
{
   fn as_matrix(self) -> DMatrix<f64>;
}

// trivial implementation
impl AsMatrix for DMatrix<f64>
{
   fn as_matrix(self) -> DMatrix<f64>
   {
      self
   }
}

// implementation on single slice

// implementation on vector of slices

// implementation on slice of slices ?

