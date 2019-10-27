use nalgebra::{DMatrix, DVector};
use crate::algebra;

//-----------------------------------------------------------------------------
// TRAITS

/// handles conversion to DMatrix type and stores information on associated output type
pub trait InputMatrix
{
   type InVector;
   type OutVector;

   // TODO add into_dmatrix and into_dvector with default implem

   /// converts an input matrix to a DMatrix
   fn to_dmatrix(m: &Self) -> DMatrix<f64>;

   /// converts an input vector to a DVector
   fn to_dvector(v: &Self::InVector) -> DVector<f64>;

   /// converts a DVector to an output vector
   fn from_dvector(v: DVector<f64>) -> Self::OutVector;
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

   /// converts a DVector to an output vector
   fn from_dvector(v: DVector<f64>) -> Self::OutVector
   {
      v
   }
}

// single row
impl InputMatrix for &[f64]
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
   fn from_dvector(v: DVector<f64>) -> Self::OutVector
   {
      assert_eq!(v.nrows(), 1);
      v[0]
   }
}

// multiple rows, base rust type
impl<'a> InputMatrix for &'a [Vec<f64>]
{
   type InVector = &'a [f64];
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
   fn from_dvector(v: DVector<f64>) -> Self::OutVector
   {
      v.iter().cloned().collect()
   }
}