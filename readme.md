# Friedrich : Gaussian Process Regression

This libarie implements [Gaussian Process Regression](https://en.wikipedia.org/wiki/Gaussian_process), also known as [Kriging](https://en.wikipedia.org/wiki/Kriging), in Rust.
Our goal is to provide a building block for other algorithms (such as [Bayesian Optimization](https://en.wikipedia.org/wiki/Bayesian_optimization)).

Gaussian process have both the ability to extract a lot of information from their training data and to return a prediction and an uncertainty on their prediction.
Furthermore, they can handle non-linear phenomenons, take uncertainty on the inputs into account and encode a prior on the output.

All of those properties make them an algorithm of choice to perform regression when data is scarce or when having uncertainty bars on the ouput is a desirable property.

However, the `o(n^3)` complexity of the algorithm makes the classical implementation unsuitable for large datasets.

**WARNING: This crate is still in an alpha state, its interface might evolve.**

## Functionalities

This implementation lets you :

- define a gaussian process with default parameters or using the builder pattern
- train it on multidimensional data
- fit the parameters (kernel and prior) on the training data
- add additional samples and refit the process
- predict the mean and variance and covariance matrix for given inputs
- sample the distribution at a given position

## Code sample

```rust
use friedrich::gaussian_process::GaussianProcess;

// trains a gaussian process on a dataset of one dimension vectors
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

- `Vec<Vec<f64>> -> Vec<f64>` each inner vector is a multidimentional training sample
- `Vec<f64> -> f64` a single multidimensional sample
- `DMatrix<f64> -> DVector<f64>` using a [nalgebra](https://www.nalgebra.org/) matrix with one row per sample

A trait is provided to add your own pairs.

## Potential future developements

The list of things that could be done to improve on the current implementation includes :

- Add better algorithms to fit kernel parameters (cross validation or gradient descent on likelyhood).
- Add [ndarray](https://docs.rs/ndarray/) support behind a feature flag.
- Add simple kernel regression (not as clever but much faster).

*Do not hesitate to send pull request or ask for features.*
