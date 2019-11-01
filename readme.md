# Friedrich : Gaussian Process Regression

This libarie implements [Gaussian Process Regression](https://en.wikipedia.org/wiki/Gaussian_process) in Rust.
Our goal is to provide a building block for other algorithms (such as [Bayesian Optimization](https://en.wikipedia.org/wiki/Bayesian_optimization)).

Gaussian process have both the ability to extract a lot of information from their training data and to return a prediction and an uncertainty on their prediction.
Furthermore, they can handle non-linear phenomenons, take uncertainty on the inputs into account and encode a prior on the output.

All of those properties make them an algorithm of choice to perform regression when data is scarce or when having uncertainty bars on the ouput is a desirable property.

However, the `o(n^3)` complexity of the algorithm makes the classical implementation unsuitable for large datasets.

## Usage

```rust
# use friedrich::gaussian_process::GaussianProcess;
# fn main() {
// trains a gaussian process on a dataset
let training_inputs = vec![vec![0.8], vec![1.2], vec![3.8], vec![4.2]];
let training_outputs = vec![3.0, 4.0, -2.0, -2.0];
let gp = GaussianProcess::default(training_inputs, training_outputs);

// predicts the mean and variance of a point
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
# }
```

## Potential future developements

The list of things that could be done to improve on the current implementation includes :

- Add better algorithms to fit kernel parameters (cross validation or gradient descent on likelyhood).
- Improve efficiency of the linear algebra operatins used.
- Add function to predict both mean and variance.
- Add [ndarray](https://docs.rs/ndarray/) support behind a feature flag.
- Add simple kernel regression (not as clever but much faster).

*Do not hesitate to send pull request or ask for features.*
