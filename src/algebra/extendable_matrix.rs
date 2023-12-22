//! Resizable matrices
//!
//! Matrix and vectors that can be extended with additional rows.

use super::{MatrixSlice, VectorSlice};
use crate::algebra::{SMatrix, SVector};
use nalgebra::*;
use nalgebra::{storage::Storage, Dynamic, U1};

//-----------------------------------------------------------------------------
// MATRIX

/// a matrix that can grow to add additional rows efficiently
#[cfg_attr(feature = "friedrich_serde", derive(serde::Deserialize, serde::Serialize))]
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

    /// Adds rows to the matrix.
    pub fn add_rows<S: Storage<f64, Dynamic, Dynamic>>(&mut self, rows: &SMatrix<S>)
    {
        // If we do not have enough rows, we grow the underlying matrix.
        let capacity = self.data.nrows();
        let required_size = self.nrows + rows.nrows();
        if required_size > capacity
        {
            // Compute new capacity.
            let growed_capacity = (3 * capacity) / 2; // capacity increased by a factor 1.5
            let new_capacity = std::cmp::max(required_size, growed_capacity);
            // Builds new matrix with more rows.
            let mut new_data = DMatrix::from_element(new_capacity, self.data.ncols(), std::f64::NAN);
            new_data.index_mut((..self.nrows, ..)).copy_from(&self.as_matrix());
            self.data = new_data;
        }

        // Add rows below data.
        self.data.index_mut((self.nrows..required_size, ..)).copy_from(rows);
        self.nrows += rows.nrows();
    }

    /// returns a slice to the data inside the extendable matrix
    pub fn as_matrix(&self) -> MatrixSlice
    {
        self.data.index((..self.nrows, ..))
    }
}

//-----------------------------------------------------------------------------
// VECTOR

/// A vector that can grow to add additional entries efficiently.
#[cfg_attr(feature = "friedrich_serde", derive(serde::Deserialize, serde::Serialize))]
pub struct EVector
{
    data: DVector<f64>,
    nrows: usize
}

impl EVector
{
    pub fn new(data: DVector<f64>) -> Self
    {
        let nrows = data.nrows();
        EVector { data, nrows }
    }

    /// Add rows to the vector.
    pub fn add_rows<S: Storage<f64, Dynamic, U1>>(&mut self, rows: &SVector<S>)
    {
        // If we do not have enough rows, we grow the underlying vector.
        let capacity = self.data.nrows();
        let required_size = self.nrows + rows.nrows();
        if required_size > capacity
        {
            // Compute new capacity.
            let growed_capacity = (3 * capacity) / 2; // capacity increased by a factor 1.5
            let new_capacity = std::cmp::max(required_size, growed_capacity);
            // Builds new matrix with more rows.
            let mut new_data = DVector::from_element(new_capacity, std::f64::NAN);
            new_data.index_mut((..self.nrows, ..)).copy_from(&self.as_vector());
            self.data = new_data;
        }

        // Add rows below data.
        self.data.index_mut((self.nrows..required_size, ..)).copy_from(rows);
        self.nrows += rows.nrows();
    }

    /// Returns a slice to the data inside the extendable matrix.
    pub fn as_vector(&self) -> VectorSlice
    {
        self.data.index((..self.nrows, ..))
    }

    /// assigns new content to the vector
    /// the new vector must be of the same size as the old vector
    pub fn assign<S: Storage<f64, Dynamic, U1>>(&mut self, rows: &SVector<S>)
    {
        assert_eq!(rows.nrows(), self.nrows);
        self.data.index_mut((..rows.nrows(), ..)).copy_from(rows);
    }
}

#[cfg(test)]
mod tests
{
    use super::*;
    use crate::conversion::InternalConvert;

    #[test]
    fn ematrix_add_rows_extend_size_when_some_of_current_data_is_masked()
    {
        let x = (vec![vec![1.0f64], vec![2.0f64]]).i_into();
        let mut e = EMatrix::new(x.clone());
        for _ in 0..5
        {
            e.add_rows(&x);
        }
    }
}
