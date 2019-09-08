//! Operations on matrix

use nalgebra::*;
use crate::parameters::kernel::Kernel;

/// represens a view to a row from a matrix
pub type RowVectorSlice<'a> = Matrix<f64, U1, Dynamic, SliceStorage<'a, f64, U1, Dynamic, U1, Dynamic>>;

/// produces a matrix full of one
pub fn one(nbrows: usize, nbcols: usize) -> DMatrix<f64>
{
   DMatrix::from_element(nbrows, nbcols, 1f64)
}

/// add the rows of the bottom matrix below the rows of the top matrix
/// NOTE: this function is more efficient if top is larger than bottom
pub fn add_rows(top: &mut DMatrix<f64>, bottom: &DMatrix<f64>)
{
   assert_eq!(top.ncols(), bottom.ncols());
   let nrows_top = top.nrows();

   // use empty temporary matrix to take ownership of top matrix in order to grow it
   let mut result = DMatrix::<f64>::zeros(0, 0);
   std::mem::swap(top, &mut result);

   // builds new matrix by adding enough rows to result matrix
   result = result.resize_vertically(bottom.nrows(), 0f64);

   // copy bottom matrix on the bottom of the new matrix
   result.index_mut((nrows_top.., ..)).copy_from(bottom);

   // put result back in top
   std::mem::swap(top, &mut result);
}

/// computes a covariance matrix using a given kernel and two matrices
/// the output has one row per row in m1 and one column per row in m2
/// TODO we could use that fact that the matrix is symmetric to reduce computations and do only lower part
pub fn make_covariance_matrix<K: Kernel>(m1: &DMatrix<f64>, m2: &DMatrix<f64>, kernel: &K) -> DMatrix<f64>
{
   return DMatrix::<f64>::from_fn(m1.nrows(), m2.nrows(), |r, c| {
      let x = m1.row(r);
      let y = m2.row(c);
      kernel.kernel(x, y)
   });
}
