use nalgebra::DVector;

/// converts some data into a valid output vector
pub trait AsVector: Sized
{
   fn as_vector(&self) -> DVector<f64>;

   fn into_vector(self) -> DVector<f64>
   {
      self.as_vector()
   }

   fn from_vector(v: DVector<f64>) -> Self;
}

/// trivial implementation for DVector
/// WARNING: this does a clone
impl AsVector for DVector<f64>
{
   fn as_vector(&self) -> DVector<f64>
   {
      self.clone()
   }

   fn into_vector(self) -> DVector<f64>
   {
      self
   }

   fn from_vector(v: DVector<f64>) -> Self
   {
      v
   }
}

/// implementation for Vec
impl AsVector for Vec<f64>
{
   fn as_vector(&self) -> DVector<f64>
   {
      DVector::from_column_slice(self)
   }

   fn from_vector(v: DVector<f64>) -> Self
   {
      v.iter().cloned().collect()
   }
}
