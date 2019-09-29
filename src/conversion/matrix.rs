use nalgebra::{DMatrix, RowDVector};

/// trait that handles convertion from arbitrary data to valid matrix
pub trait AsMatrix: Sized
{
   /// converts a reference to a matrix
   fn as_matrix(&self) -> DMatrix<f64>;

   /// converts a value into a matrix using faster methods if possible
   fn into_matrix(self) -> DMatrix<f64>
   {
      self.as_matrix()
   }
}

/// trivial implementation for DMatrix type
impl AsMatrix for DMatrix<f64>
{
   fn as_matrix(&self) -> DMatrix<f64>
   {
      self.clone()
   }

   fn into_matrix(self) -> DMatrix<f64>
   {
      self
   }
}

/// from a Vec of Vec in row major disposition
impl AsMatrix for Vec<Vec<f64>>
{
   fn as_matrix(&self) -> DMatrix<f64>
   {
      let rows: Vec<RowDVector<f64>> = self.iter().map(|v| RowDVector::from_row_slice(v)).collect();
      DMatrix::from_rows(&rows)
   }
}

/// implementation for single column
impl AsMatrix for Vec<f64>
{
   fn as_matrix(&self) -> DMatrix<f64>
   {
      DMatrix::from_column_slice(self.len(), 1, self)
   }
}
