
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
// MATRIX REF

/// a matrix out of which a slice can be extracted
pub trait InputRef 
{
   /// returns a slice that points to the content of the matrix
   fn to_mslice(&self) -> MatrixSlice;
}

impl InputRef for DMatrix<f64>
{
   fn to_mslice(&self) -> MatrixSlice
   {
      self.index((.., ..))
   }
}

//-----------------------------------------------------------------------------
// VECTOR REF

/// a vector out of which a slice can be extracted
pub trait OutputRef
{
   /// returns a slice that points to the content of the vector
   fn to_vslice(&self) -> VectorSlice;
}

impl OutputRef for DVector<f64>
{
   fn to_vslice(&self) -> VectorSlice
   {
      self.index((.., ..))
   }
}