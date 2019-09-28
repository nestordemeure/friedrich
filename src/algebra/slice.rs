
use nalgebra::*;

//-----------------------------------------------------------------------------
// TYPES

/// represents a view to a row from a matrix
pub type RowVectorSlice<'a> = Matrix<f64, U1, Dynamic, SliceStorage<'a, f64, U1, Dynamic, U1, Dynamic>>;

/// represents a view to a column from a matrix
pub type VectorSlice<'a> = Matrix<f64, Dynamic, U1, SliceStorage<'a, f64, Dynamic, U1, U1, Dynamic>>;

/// represents a slice of a matrix
pub type MatrixSlice<'a> = Matrix<f64, Dynamic, Dynamic, SliceStorage<'a, f64, Dynamic, Dynamic, U1, Dynamic>>;

//-----------------------------------------------------------------------------
// CONVERSION

/// a matrix out of which a slice can be extracted
pub trait SliceableMatrix 
{
   /// returns a slice that points to the content of the matrix
   fn data(&self) -> MatrixSlice;
}

impl SliceableMatrix for DMatrix<f64>
{
   fn data(&self) -> MatrixSlice
   {
      self.index((.., ..))
   }
}