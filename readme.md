# Friedrich : Gaussian Process Regression

[![Crates.io](https://img.shields.io/crates/v/friedrich)](https://crates.io/crates/friedrich)
[![docs](https://img.shields.io/crates/v/friedrich?color=yellow&label=docs)](https://docs.rs/friedrich)

This library implements [Gaussian Process Regression](https://en.wikipedia.org/wiki/Gaussian_process), also known as [Kriging](https://en.wikipedia.org/wiki/Kriging), in Rust.
Our goal is to provide a solid and well featured building block for other algorithms (such as [Bayesian Optimization](https://en.wikipedia.org/wiki/Bayesian_optimization)).

Gaussian processes have both the ability to extract a lot of information from their training data and to return a prediction and an uncertainty value on their prediction.
Furthermore, they can handle non-linear phenomena, take uncertainty on the inputs into account, and encode a prior on the output.

All of those properties make it an algorithm of choice to perform regression when data is scarce or when having uncertainty bars on the ouput is a desirable property.

However, the `O(n^3)` complexity of the algorithm makes the classic implementations unsuitable for large datasets.

## Functionalities

This implementation lets you:

- define a gaussian process with default parameters or using the builder pattern
- train it on multidimensional data
- fit the parameters (kernel, prior and noise) on the training data
- add additional samples efficiently (`O(n^2)`) and refit the process
- predict the mean, variance and covariance matrix for given inputs
- sample the distribution at a given position
- save and load a trained model with [serde](https://serde.rs/)

(See the [todo](https://github.com/nestordemeure/friedrich/blob/master/todo.md) file to get up-to-date informations on current developements.)

## Code sample

```rust
use friedrich::gaussian_process::GaussianProcess;

// trains a gaussian process on a dataset of one-dimensional vectors
let training_inputs = vec![vec![0.8], vec![1.2], vec![3.8], vec![4.2]];
let training_outputs = vec![3.0, 4.0, -2.0, -2.0];
let gp = GaussianProcess::default(training_inputs, training_outputs);

// predicts the mean and variance of a single point
let input = vec![1.];
let mean = gp.predict(&input);
let var = gp.predict_variance(&input);
println!("prediction: {} ± {}", mean, var.sqrt());

// makes several prediction
let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
let outputs = gp.predict(&inputs);
println!("predictions: {:?}", outputs);

// samples from the distribution
let new_inputs = vec![vec![1.0], vec![2.0]];
let sampler = gp.sample_at(&new_inputs);
let mut rng = rand::thread_rng();
println!("samples: {:?}", sampler.sample(&mut rng));
```

## Inputs

Most methods of this library can currently work with the following `input -> ouput` pairs :

- `Vec<f64> -> f64` a single, multidimensional, sample
- `Vec<Vec<f64>> -> Vec<f64>` each inner vector is a training sample
- `DMatrix<f64> -> DVector<f64>` using a [nalgebra](https://www.nalgebra.org/) matrix with one row per sample
- `ArrayBase<f64, Ix1> -> f64` a single sample stored in a [ndarray](https://crates.io/crates/ndarray) array (using the `friedrich_ndarray` feature)
- `ArrayBase<f64, Ix2> -> Array1<f64>` each row is a sample (using the `friedrich_ndarray` feature)

The [Input trait](https://docs.rs/friedrich/latest/friedrich/trait.Input.html) is provided to add your own pairs.

## Why call it Friedrich ?

Gaussian Process are named after the [Gaussian distribution](https://en.wikipedia.org/wiki/Gaussian_function) which is itself named after [Carl Friedrich Gauss](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss).
