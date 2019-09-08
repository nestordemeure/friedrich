//! Operations on matrix

use nalgebra::{DMatrix, DVector};
use crate::parameters::kernel::Kernel;

/// add the rows of the bottom matrix below the rows of the top matrix
/// NOTE: this function is more efficient if top is larger than bottom
pub fn add_rows(top: &mut DMatrix<f64>, bottom: &DMatrix<f64>)
{
   assert_eq!(top.ncols(), bottom.ncols());
   let nrows_top = top.nrows();
   let nrows_bottom = bottom.nrows();

   // use empty temporary matrix to take ownership of top matrix in order to grow it
   let mut result = DMatrix::<f64>::zeros(0, 0);
   std::mem::swap(top, &mut result);

   // builds new matrix by adding enough rows to result matrix
   result = result.resize_vertically(nrows_bottom, 0f64);

   // copy bottom matrix on the bottom of the new matrix
   // TODO use copy_from instead of loop
   // https://discourse.nphysics.org/t/creating-block-sparse-matrices/401/5
   for col in 0..bottom.ncols()
   {
      for row in 0..nrows_bottom
      {
         result[(nrows_top + row, col)] = bottom[(row, col)];
      }
   }

   // put result back in top
   std::mem::swap(top, &mut result);
}

/// computes a covariance matrix using a given kernel
pub fn make_covariance_matrix<K:Kernel>(m1: &DMatrix<f64>, m2: &DMatrix<f64>, kernel: &K) -> DMatrix<f64>
{
   return DMatrix::<f64>::from_fn(m1.nrows(), m2.nrows(), |r,c| 
   {
      let x = m1.row(r);
      let y = m2.row(c);
      kernel.kernel(x, y)
   })
}
