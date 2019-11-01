use nalgebra::{DMatrix, DVector};
use crate::algebra;

//-----------------------------------------------------------------------------
// TRAITS

/// Add Input type -> Output type pairs
///
/// Handles conversion to DMatrix type and stores information on associated output type.
///
/// User-defined input type should implement this trait.
pub trait Input: Sized
{
   /// type of the vectors storing training output data and given to methods
   type InVector: Sized;
   /// type of the vectors outputed when a method is called
   type OutVector;

   /// Converts an input matrix to a DMatrix.
   fn to_dmatrix(m: &Self) -> DMatrix<f64>;

   /// Optional: converts an owned input matrix to a DMatrix.
   /// This function is used for to reduce copies when the input type is compatible with DMatrix.
   fn into_dmatrix(m: Self) -> DMatrix<f64>
   {
      Self::to_dmatrix(&m)
   }

   /// Converts an input vector to a DVector.
   fn to_dvector(v: &Self::InVector) -> DVector<f64>;

   /// converts an input vector to a DVector.
   fn into_dvector(v: Self::InVector) -> DVector<f64>
   {
      Self::to_dvector(&v)
   }

   /// converts a DVector to an output vector.
   fn from_dvector(v: &DVector<f64>) -> Self::OutVector;
}

//-----------------------------------------------------------------------------
// IMPLEMENTATIONS

/// direct implementation
impl Input for DMatrix<f64>
{
   type InVector = DVector<f64>;
   type OutVector = DVector<f64>;

   /// converts an input matrix to a DMatrix
   fn to_dmatrix(m: &Self) -> DMatrix<f64>
   {
      m.clone()
   }

   /// converts an input vector to a DVector
   fn to_dvector(v: &Self::InVector) -> DVector<f64>
   {
      v.clone()
   }

   /// converts an input matrix to a DMatrix
   fn into_dmatrix(m: Self) -> DMatrix<f64>
   {
      m
   }

   /// converts an input vector to a DVector
   fn into_dvector(v: Self::InVector) -> DVector<f64>
   {
      v
   }

   /// converts a DVector to an output vector
   fn from_dvector(v: &DVector<f64>) -> Self::OutVector
   {
      v.clone()
   }
}

/// single row
impl Input for Vec<f64>
{
   type InVector = f64;
   type OutVector = f64;

   /// converts an input matrix to a DMatrix
   fn to_dmatrix(m: &Self) -> DMatrix<f64>
   {
      DMatrix::from_row_slice(1, m.len(), m)
   }

   /// converts an input vector to a DVector
   fn to_dvector(v: &Self::InVector) -> DVector<f64>
   {
      DVector::from_element(1, *v)
   }

   /// converts a DVector to an output vector
   fn from_dvector(v: &DVector<f64>) -> Self::OutVector
   {
      assert_eq!(v.nrows(), 1);
      v[0]
   }
}

/// multiple rows, base rust type
impl Input for Vec<Vec<f64>>
{
   type InVector = Vec<f64>;
   type OutVector = Vec<f64>;

   /// converts an input matrix to a DMatrix
   fn to_dmatrix(m: &Self) -> DMatrix<f64>
   {
      algebra::make_matrix_from_row_slices(m)
   }

   /// converts an input vector to a DVector
   fn to_dvector(v: &Self::InVector) -> DVector<f64>
   {
      DVector::from_column_slice(v)
   }

   /// converts a DVector to an output vector
   fn from_dvector(v: &DVector<f64>) -> Self::OutVector
   {
      v.iter().cloned().collect()
   }
}
