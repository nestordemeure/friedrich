# Gaussian Process

**This is a work in progress!**
This crate is still in its alpha stage, the interface and internals might still evolve a lot in the following weeks.

My aim is to build a versatile and, hopefully, scalable implementation of [Gaussian Process Regression](https://en.wikipedia.org/wiki/Gaussian_process) in Rust.

## Usage

The algorithm works on matrices (see [nalgebra](https://www.nalgebra.org/quick_reference/)) of inputs / outputs.

## TODO

- Clean-up the documentation.
- Make the implementation usable wit other number types such as `f32` ans `complex<f64>`.
- Add better algorithms to fit parameters.

- add function to make prediction on single sample
- encode expected output type in GP type ? or default to vector ?
