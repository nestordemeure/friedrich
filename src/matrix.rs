//! Operations on matrix

use nalgebra::{DMatrix};

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
