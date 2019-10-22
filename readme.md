# Gaussian Process

**This is a work in progress!**

This crate is still in its alpha stage, the interface and internals might still evolve a lot.

My aim is to implement [Gaussian Process Regression](https://en.wikipedia.org/wiki/Gaussian_process) in Rust.

## Usage

The algorithm works on matrices (see [nalgebra](https://www.nalgebra.org/quick_reference/)) of inputs / outputs.

## TODO

- Clean-up the documentation.
- Add better algorithms to fit kernel parameters.
- Add ndarray support behind a feature flag

add trait IntoDVector
and have only one implementation that relies on it
but then how to know what will the output type be... (we fall on out previous problems...)
we could have type that implement inputMatrix and have a linked input vector type

3 traits:
FromDVector: implemented by output types
InputVector: implemented by input type
InputMatrix: implemented by input type, includes an output type encoded

with that we can fuse the several, the nalgebra and the base
but it might make the interface less readable... it will need testing in a separate branch