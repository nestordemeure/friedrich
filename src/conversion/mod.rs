use nalgebra::{DMatrix, DVector};

#[cfg(feature = "friedrich_ndarray")]
use ndarray::{Array1, ArrayBase, Data, Ix1, Ix2};

//-----------------------------------------------------------------------------
// TRAITS

/// Implemented by `Input -> Output` type pairs
///
/// Handles conversion to DMatrix type and stores information on associated output type.
/// Most methods of this library can currently work with the following `input -> ouput` pairs :
///
/// - `Vec<f64> -> f64` a single, multidimensional, sample
/// - `Vec<Vec<f64>> -> Vec<f64>` each inner vector is a training sample
/// - `DMatrix<f64> -> DVector<f64>` using a [nalgebra](https://www.nalgebra.org/) matrix with one row per sample
/// - `ArrayBase<f64, Ix1> -> f64` a single sample stored in a [ndarray](https://crates.io/crates/ndarray) array (using the `friedrich_ndarray` feature)
/// - `ArrayBase<f64, Ix2> -> Array1<f64>` each row is a sample (using the `friedrich_ndarray` feature)
///
/// User-defined input type should implement this trait.
pub trait Input: Sized {
    /// type of the vectors storing training output data and given to methods
    type InVector: Sized;
    /// type of the vectors outputed when a method is called
    type OutVector;

    /// Converts an input matrix to a DMatrix.
    fn to_dmatrix(m: &Self) -> DMatrix<f64>;

    /// Optional: converts an owned input matrix to a DMatrix.
    /// This function is used for to reduce copies when the input type is compatible with DMatrix.
    fn into_dmatrix(m: Self) -> DMatrix<f64> {
        Self::to_dmatrix(&m)
    }

    /// Converts an input vector to a DVector.
    fn to_dvector(v: &Self::InVector) -> DVector<f64>;

    /// Optional: converts an owned input vector to a DVector.
    /// This function is used for to reduce copies when the input type is compatible with DVector.
    fn into_dvector(v: Self::InVector) -> DVector<f64> {
        Self::to_dvector(&v)
    }

    /// converts a DVector to an output vector.
    fn from_dvector(v: &DVector<f64>) -> Self::OutVector;
}

//-----------------------------------------------------------------------------
// IMPLEMENTATIONS

/// direct implementation
impl Input for DMatrix<f64> {
    type InVector = DVector<f64>;
    type OutVector = DVector<f64>;

    /// converts an input matrix to a DMatrix
    fn to_dmatrix(m: &Self) -> DMatrix<f64> {
        m.clone()
    }

    /// converts an input vector to a DVector
    fn to_dvector(v: &Self::InVector) -> DVector<f64> {
        v.clone()
    }

    /// converts an input matrix to a DMatrix
    fn into_dmatrix(m: Self) -> DMatrix<f64> {
        m
    }

    /// converts an input vector to a DVector
    fn into_dvector(v: Self::InVector) -> DVector<f64> {
        v
    }

    /// converts a DVector to an output vector
    fn from_dvector(v: &DVector<f64>) -> Self::OutVector {
        v.clone()
    }
}

/// single row
impl Input for Vec<f64> {
    type InVector = f64;
    type OutVector = f64;

    /// converts an input matrix to a DMatrix
    fn to_dmatrix(m: &Self) -> DMatrix<f64> {
        DMatrix::from_row_slice(1, m.len(), m)
    }

    /// converts an input vector to a DVector
    fn to_dvector(v: &Self::InVector) -> DVector<f64> {
        DVector::from_element(1, *v)
    }

    /// converts a DVector to an output vector
    fn from_dvector(v: &DVector<f64>) -> Self::OutVector {
        assert_eq!(v.nrows(), 1);
        v[0]
    }
}

/// multiple rows, base rust type
impl Input for Vec<Vec<f64>> {
    type InVector = Vec<f64>;
    type OutVector = Vec<f64>;

    /// converts an input matrix to a DMatrix
    fn to_dmatrix(m: &Self) -> DMatrix<f64> {
        let nb_rows = m.len();
        assert_ne!(nb_rows, 0);
        let nb_cols = m[0].len();
        DMatrix::from_fn(nb_rows, nb_cols, |r, c| m[r][c])
    }

    /// converts an input vector to a DVector
    fn to_dvector(v: &Self::InVector) -> DVector<f64> {
        DVector::from_column_slice(v)
    }

    /// converts a DVector to an output vector
    fn from_dvector(v: &DVector<f64>) -> Self::OutVector {
        v.iter().cloned().collect()
    }
}

/// multiple rows, ndarray array type
#[cfg(feature = "friedrich_ndarray")]
impl<D: Data<Elem = f64>> Input for ArrayBase<D, Ix2> {
    type InVector = ArrayBase<D, Ix1>;
    type OutVector = Array1<f64>;

    /// converts an input matrix to a DMatrix
    fn to_dmatrix(m: &Self) -> DMatrix<f64> {
        assert_ne!(m.nrows(), 0);
        // use `.t()` to get from row-major to col-major
        DMatrix::from_iterator(m.nrows(), m.ncols(), m.t().iter().cloned())
    }

    /// converts an input vector to a DVector
    fn to_dvector(v: &Self::InVector) -> DVector<f64> {
        DVector::from_iterator(v.len(), v.iter().cloned())
    }

    /// converts a DVector to an output vector
    fn from_dvector(v: &DVector<f64>) -> Self::OutVector {
        v.iter().cloned().collect()
    }
}

/// single row, ndarray array type
#[cfg(feature = "friedrich_ndarray")]
impl<D: Data<Elem = f64>> Input for ArrayBase<D, Ix1> {
    type InVector = f64;
    type OutVector = f64;

    /// converts an input matrix to a DMatrix
    fn to_dmatrix(m: &Self) -> DMatrix<f64> {
        DMatrix::from_iterator(1, m.len(), m.iter().cloned())
    }

    /// converts an input vector to a DVector
    fn to_dvector(v: &Self::InVector) -> DVector<f64> {
        DVector::from_element(1, *v)
    }

    /// converts a DVector to an output vector
    fn from_dvector(v: &DVector<f64>) -> Self::OutVector {
        assert_eq!(v.nrows(), 1);
        v[0]
    }
}
