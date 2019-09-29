//! Operations on matrix
mod extendable_matrix;

use nalgebra::*;
use crate::parameters::kernel::Kernel;
use crate::conversion::MatrixSlice;
pub use extendable_matrix::{EMatrix, EVector};

//-----------------------------------------------------------------------------
// COVARIANCE MATRIX

/// computes a covariance matrix using a given kernel and two matrices
/// the output has one row per row in m1 and one column per row in m2
pub fn make_covariance_matrix<K: Kernel>(m1: MatrixSlice, m2: MatrixSlice, kernel: &K) -> DMatrix<f64>
{
   return DMatrix::<f64>::from_fn(m1.nrows(), m2.nrows(), |r, c| {
      let x = m1.row(r);
      let y = m2.row(c);
      kernel.kernel(x, y)
   });
}

/// computes the cholesky decomposition of the covariance matrix of some inputs
/// adds a given diagonal noise
/// relies on the fact that only the lower triangular part of the matrix is needed for the decomposition
pub fn make_cholesky_covariance_matrix<K: Kernel>(inputs: MatrixSlice,
                                                  kernel: &K,
                                                  diagonal_noise: f64)
                                                  -> Cholesky<f64, Dynamic>
{
   // empty covariance matrix
   // TODO it would be faster to start with an an uninitialized matrix but it would require unsafe
   let mut covmatix = DMatrix::<f64>::from_element(inputs.nrows(), inputs.nrows(), std::f64::NAN);

   // computes the covariance for all the lower triangular matrix
   for (col_index, x) in inputs.row_iter().enumerate()
   {
      for (row_index, y) in inputs.row_iter().enumerate().skip(col_index)
      {
         covmatix[(row_index, col_index)] = kernel.kernel(x, y);
      }

      // adds diagonal noise
      covmatix[(col_index, col_index)] += diagonal_noise * diagonal_noise;
   }

   return covmatix.cholesky().expect("Cholesky decomposition failed!");
}
