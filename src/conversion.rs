/// Implemented by `Input -> Output` type pairs
///
/// Handles conversion to DMatrix type and stores information on associated output type.
/// Most methods of this library can currently work with the following `input -> output` pairs :
///
/// Input | Output | Description
/// ---|---|---
/// [`Array2<f64>`](https://docs.rs/ndarray/0.15/ndarray/type.Array2.html) | [`Array1<f64>`](https://docs.rs/ndarray/0.15/ndarray/type.Array1.html) | Multiple input vectors to multiple output values (with `friedrich_ndarray` feature).
/// [`Array1<f64>`](https://docs.rs/ndarray/0.15/ndarray/type.Array1.html) | [`f64`] | A single input vector to a single output value (with `friedrich_ndarray` feature).
/// [`DMatrix<f64>`](https://docs.rs/nalgebra/0.29/nalgebra/base/type.DMatrix.html) | [`DVector<f64>`](https://docs.rs/nalgebra/0.29/nalgebra/base/type.DVector.html) | Multiple input vectors to multiple output values.
/// [`DVector<f64>`](https://docs.rs/nalgebra/0.29/nalgebra/base/type.DVector.html) | [`f64`] | A single input vector to a single output value.
/// [`Vec<Vec<f64>>`] | [`Vec<f64>` ] | Multiple input vectors to multiple output values.
/// [`Vec<f64>`] | [`f64` ] | A single input vector to a single input value.
///
/// User-defined input type should implement this trait.
pub trait Input: InternalConvert<nalgebra::base::DMatrix<f64>> + Clone
{
    type Output: InternalConvert<nalgebra::base::DVector<f64>> + Clone;
}
/// We cannot implement a foreign trait on foreign types, so we use this local trait to mimic [`std::convert::From`].
pub trait InternalConvert<T>
{
    fn i_into(self) -> T;
    fn i_from(x: T) -> Self;
}
// Implementing for ndarray
// -------------------------------------------
#[cfg(feature = "friedrich_ndarray")]
impl Input for ndarray::Array1<f64>
{
    type Output = f64;
}
#[cfg(feature = "friedrich_ndarray")]
impl InternalConvert<nalgebra::base::DMatrix<f64>> for ndarray::Array1<f64>
{
    fn i_into(self) -> nalgebra::base::DMatrix<f64>
    {
        nalgebra::base::DMatrix::from_iterator(1, self.len(), self.into_iter())
    }
    fn i_from(x: nalgebra::base::DMatrix<f64>) -> Self
    {
        Self::from_iter(x.into_iter())
    }
}
// -------------------------------------------
#[cfg(feature = "friedrich_ndarray")]
impl Input for ndarray::Array2<f64>
{
    type Output = ndarray::Array1<f64>;
}
#[cfg(feature = "friedrich_ndarray")]
impl InternalConvert<nalgebra::base::DMatrix<f64>> for ndarray::Array2<f64>
{
    fn i_into(self) -> nalgebra::base::DMatrix<f64>
    {
        nalgebra::base::DMatrix::from_iterator(self.dim().0, self.dim().1, self.into_iter())
    }
    fn i_from(x: nalgebra::base::DMatrix<f64>) -> Self
    {
        Self::from_shape_vec(x.shape(), x.into_iter().collect::<Vec<_>>())
    }
}
#[cfg(feature = "friedrich_ndarray")]
impl InternalConvert<nalgebra::base::DVector<f64>> for ndarray::Array1<f64>
{
    fn i_into(self) -> nalgebra::base::DVector<f64>
    {
        nalgebra::base::DVector::from_iterator(self.len(), self.into_iter())
    }
    fn i_from(x: nalgebra::base::DVector<f64>) -> Self
    {
        Self::from_iter(x.into_iter())
    }
}
// Implementing for nalgebra
// -------------------------------------------=
impl Input for nalgebra::base::DVector<f64>
{
    type Output = f64;
}
impl InternalConvert<nalgebra::base::DMatrix<f64>> for nalgebra::base::DVector<f64>
{
    fn i_into(self) -> nalgebra::base::DMatrix<f64>
    {
        nalgebra::base::DMatrix::from_iterator(1, self.len(), self.iter().cloned())
    }
    fn i_from(x: nalgebra::base::DMatrix<f64>) -> Self
    {
        Self::from_iterator(x.len(), x.into_iter().cloned())
    }
}
impl InternalConvert<nalgebra::base::DVector<f64>> for f64
{
    fn i_into(self) -> nalgebra::base::DVector<f64>
    {
        nalgebra::dvector![self]
    }
    fn i_from(x: nalgebra::base::DVector<f64>) -> Self
    {
        x[0]
    }
}
// -------------------------------------------
impl Input for nalgebra::base::DMatrix<f64>
{
    type Output = nalgebra::base::DVector<f64>;
}
impl InternalConvert<nalgebra::base::DMatrix<f64>> for nalgebra::base::DMatrix<f64>
{
    fn i_into(self) -> nalgebra::base::DMatrix<f64>
    {
        self
    }
    fn i_from(x: nalgebra::base::DMatrix<f64>) -> Self
    {
        x
    }
}
impl InternalConvert<nalgebra::base::DVector<f64>> for nalgebra::base::DVector<f64>
{
    fn i_into(self) -> nalgebra::base::DVector<f64>
    {
        self
    }
    fn i_from(x: nalgebra::base::DVector<f64>) -> Self
    {
        x
    }
}

// Implementing for vec
// -------------------------------------------
impl Input for Vec<Vec<f64>>
{
    type Output = Vec<f64>;
}
impl InternalConvert<nalgebra::base::DMatrix<f64>> for Vec<Vec<f64>>
{
    fn i_into(self) -> nalgebra::base::DMatrix<f64>
    {
        assert!(!self.is_empty());
        assert!(!self[0].is_empty());

        nalgebra::base::DMatrix::from_vec(self.len(),
                                          self[0].len(),
                                          self.into_iter().flatten().collect::<Vec<_>>())
    }
    fn i_from(x: nalgebra::base::DMatrix<f64>) -> Self
    {
        x.row_iter().map(|row| row.iter().cloned().collect::<Vec<_>>()).collect::<Vec<_>>()
    }
}
impl InternalConvert<nalgebra::base::DVector<f64>> for Vec<f64>
{
    fn i_into(self) -> nalgebra::base::DVector<f64>
    {
        nalgebra::base::DVector::from_vec(self)
    }
    fn i_from(x: nalgebra::base::DVector<f64>) -> Self
    {
        x.iter().cloned().collect::<Vec<_>>()
    }
}
// -------------------------------------------
impl Input for Vec<f64>
{
    type Output = f64;
}
impl InternalConvert<nalgebra::base::DMatrix<f64>> for Vec<f64>
{
    fn i_into(self) -> nalgebra::base::DMatrix<f64>
    {
        assert!(!self.is_empty());

        nalgebra::base::DMatrix::from_vec(self.len(), 1, self)
    }
    fn i_from(x: nalgebra::base::DMatrix<f64>) -> Self
    {
        x.iter().cloned().collect::<Vec<_>>()
    }
}
