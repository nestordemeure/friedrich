# Gaussian Process

**This is a work in progress!**

This crate is still in its alpha stage, the interface and internals might still evolve a lot.

My aim is to implement [Gaussian Process Regression](https://en.wikipedia.org/wiki/Gaussian_process) in Rust.

## Usage

The algorithm works on matrices (see [nalgebra](https://www.nalgebra.org/quick_reference/)) of inputs / outputs.

## TODO

- Clean-up the documentation.
- Add better algorithms to fit kernel parameters.
- Decide on a single input format to avoid template parameters ?
- Make the implementation usable wit other number types such as `f32` and `complex<f64>`.
- fuse gaussian process and gaussian process trained