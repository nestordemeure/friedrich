use nalgebra::DVector;

/// converts some data into a valid output vector
pub trait AsVector: Sized
{
   fn as_vector(self) -> DVector<f64>;
}

// trivial implementation
impl AsVector for DVector<f64>
{
   fn as_vector(self) -> DVector<f64>
   {
      self
   }
}

// trivial implementation
impl AsVector for &DVector<f64>
{
   fn as_vector(self) -> DVector<f64>
   {
      self.clone()
   }
}

// implementation on Rust std::Vec

// implementation on slice

// implementation on single number
