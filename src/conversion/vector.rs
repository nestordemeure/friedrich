use nalgebra::DVector;

/// converts some data into a valid output vector
pub trait AsVector: Sized
{
   fn as_vector(self) -> DVector<f64>;
}

// trivial implementation for DVector
impl AsVector for DVector<f64>
{
   fn as_vector(self) -> DVector<f64>
   {
      self
   }
}

// implementation for &DVector
impl AsVector for &DVector<f64>
{
   fn as_vector(self) -> DVector<f64>
   {
      self.clone()
   }
}

/// implementation for Vec
impl AsVector for &Vec<f64>
{
   fn as_vector(self) -> DVector<f64>
   {
      DVector::from_column_slice(self)
   }
}

/// implementation for slice
impl AsVector for &[f64]
{
   fn as_vector(self) -> DVector<f64>
   {
      DVector::from_column_slice(self)
   }
}