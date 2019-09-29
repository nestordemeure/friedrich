mod matrix;
mod vector;
mod slice;

pub use matrix::AsMatrix;
pub use vector::AsVector;
pub use slice::{MatrixSlice, RowVectorSlice, VectorSlice, InputRef, OutputRef};

/*
   traits:
   self -> AsMatrix
   self -> AsVector
   &self -> AsMatrixSlice
   &self -> AsVectorSlice
*/
