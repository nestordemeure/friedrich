use nalgebra::DMatrix;

/// trait that handles convertion from arbitrary data to valid matrix
pub trait AsMatrix: Sized
{
   /// converts a reference to a matrix
   fn as_matrix(self) -> DMatrix<f64>;
}

impl AsMatrix for DMatrix<f64>
{
   fn as_matrix(self) -> DMatrix<f64>
   {
      self
   }
}

impl AsMatrix for &DMatrix<f64>
{
   fn as_matrix(self) -> DMatrix<f64>
   {
      self.clone()
   }
}

// implementation on single slice

// implementation on vector of slices

// implementation on slice of slices ?
