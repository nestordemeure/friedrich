# Gaussian Process

**This is a work in progress!**
This crate is still in its alpha stage, the interface and internals might still evolve a lot in the following weeks.

My aim is to build a versatile and, hopefully, scalable implementation of [Gaussian Process Regression](https://en.wikipedia.org/wiki/Gaussian_process) in Rust.

## Usage

The algorithm works on matrices (see [nalgebra](https://www.nalgebra.org/quick_reference/)) of inputs / outputs.

## TODO

- Make the implementation usable wit other number types such as `f32` ans `complex<f64>`.
- Make it possible to give sample with other format. I could take `Vec<slice>` by default as inputs (or implement an implicit cast from `Vec<slice>` and `Vec<f64>` to `DMatrix`).
- Make the covariance function output a structure that can be used to sample from the model.
- Clean-up the documentation.
- Add better algorithms to fit parameters.
