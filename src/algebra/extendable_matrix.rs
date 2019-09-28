use nalgebra::*;
use super::{MatrixSlice, SliceableMatrix};

/// a matrix that can grow to add additional rows efficiently
pub struct EMatrix
{
   data: DMatrix<f64>,
   nrows: usize
}

impl EMatrix
{
   pub fn new(data: DMatrix<f64>) -> Self
   {
      let nrows = data.nrows();
      EMatrix { data, nrows }
   }

   /// add rows to the matrix
   pub fn add_rows(&mut self, rows: &DMatrix<f64>)
   {
      // if we do not have enough rows, we grow the underlying matrix
      let capacity = self.data.nrows();
      let required_size = self.nrows + rows.nrows();
      if required_size > capacity
      {
         // compute new capacity
         let growed_capacity = (3 * capacity) / 2; // capacity increased by a factor 1.5
         let new_capacity = std::cmp::max(required_size, growed_capacity);
         // builds new matrix with more rows
         let mut new_data = DMatrix::from_element(new_capacity, self.data.ncols(), std::f64::NAN);
         new_data.index_mut((..self.nrows, ..)).copy_from(&self.data);
         self.data = new_data;
      }

      // add rows below data
      self.data.index_mut((self.nrows.., ..)).copy_from(rows);
      self.nrows += rows.nrows();
   }
}

/// converts a ref to an extendable matrix to a slice that points to the actual data
impl SliceableMatrix for EMatrix
{
      /// converts a ref to an extendable matrix to a slice that points to the actual data
   fn data(&self) -> MatrixSlice
   {
      self.data.index((..self.nrows, ..))
   }
}
