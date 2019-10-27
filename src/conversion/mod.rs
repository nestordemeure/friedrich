use nalgebra::{DMatrix, DVector};
use crate::algebra;

//-----------------------------------------------------------------------------
// TRAITS

/// handles conversion to DMatrix type and stores information on associated output type
pub trait InputMatrix: Sized
{
   type InVector: Sized;
   type OutVector;

   /// converts an input matrix to a DMatrix
   fn to_dmatrix(m: &Self) -> DMatrix<f64>;

   /// converts an input vector to a DVector
   fn to_dvector(v: &Self::InVector) -> DVector<f64>;

   /// converts an input matrix to a DMatrix
   fn into_dmatrix(m: Self) -> DMatrix<f64>
   {
      Self::to_dmatrix(&m)
   }

   /// converts an input vector to a DVector
   fn into_dvector(v: Self::InVector) -> DVector<f64>
   {
      Self::to_dvector(&v)
   }

   /// converts a DVector to an output vector
   fn from_dvector(v: &DVector<f64>) -> Self::OutVector;
}

//-----------------------------------------------------------------------------
// IMPLEMENTATIONS

// default implementation
impl InputMatrix for DMatrix<f64>
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

// single row
impl InputMatrix for Vec<f64>
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

// multiple rows, base rust type
impl InputMatrix for Vec<Vec<f64>>
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
