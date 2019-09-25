# Gaussian Process

**This is a work in progress!**
This crate is still in its alpha stage, the interface and internals might still evolve a lot in the following weeks.

My aim is to build a versatile and, hopefully, scalable implementation of [Gaussian Process Regression](https://en.wikipedia.org/wiki/Gaussian_process) in Rust.

## Usage

The algorithm works on matrices (see [nalgebra](https://www.nalgebra.org/quick_reference/)) of inputs / outputs.

## TODO

- add computation of covmatrix between output dim (=> remove amplitude fit from kernel and add it to variance computation)
- Clean-up the documentation.
- Make it possible to give sample with other format. I could take `Vec<slice>` by default as inputs (or implement an implicit cast from `Vec<slice>` and `Vec<f64>` to `DMatrix`).
- Make the implementation usable wit other number types such as `f32` ans `complex<f64>`.
- Add better algorithms to fit parameters.
