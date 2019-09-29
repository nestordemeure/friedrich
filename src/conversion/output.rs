use nalgebra::DVector;

/// converts some data into a valid output vector
pub trait Output: Sized
{
   fn to_output(self) -> DVector<f64>;
}

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
