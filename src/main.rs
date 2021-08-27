#![allow(dead_code)]

mod algebra;
mod conversion;
mod gaussian_process;
mod parameters;

use crate::gaussian_process::GaussianProcess;

fn main() {
    {
        // trains a gaussian process on a dataset of one dimension vectors
        let training_inputs = vec![vec![0.8], vec![1.2], vec![3.8], vec![4.2]];
        let training_outputs = vec![3.0, 4.0, -2.0, -2.0];
        let mut gp = GaussianProcess::default(training_inputs, training_outputs);

        // predicts the mean and variance of a single point
        let input = vec![1.];
        let mean = gp.predict(&input);
        let var = gp.predict_variance(&input);
        println!("prediction: {} ± {}", mean, var.sqrt());

        // computes the likelihood of the model
        let likelihood = gp.likelihood();
        println!("likelihood of the current model : {}", likelihood);

        // updates the model
        let additional_inputs = vec![vec![0.], vec![1.], vec![2.], vec![5.]];
        let additional_outputs = vec![2.0, 3.0, -1.0, -2.0];
        let fit_prior = true;
        let fit_kernel = true;
        let max_iter = 100;
        let convergence_fraction = 0.05;
        gp.add_samples(&additional_inputs, &additional_outputs);
        gp.fit_parameters(fit_prior, fit_kernel, max_iter, convergence_fraction);

        // makes several prediction
        let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
        let outputs = gp.predict(&inputs);
        println!("predictions: {:?}", outputs);

        // samples from the distribution
        let new_inputs = vec![vec![1.0], vec![2.0]];
        let sampler = gp.sample_at(&new_inputs);
        let mut rng = rand::thread_rng();
        for i in 1..=5 {
            println!("sample {} : {:?}", i, sampler.sample(&mut rng));
        }
    }

    {
        // trains a gaussian process on a dataset
        let training_inputs = vec![
            vec![0.8, 0.1],
            vec![1.2, 0.2],
            vec![3.8, 0.3],
            vec![4.2, 0.5],
        ];
        let training_outputs = vec![3.0, 4.0, -2.0, -2.0];
        let gp = GaussianProcess::default(training_inputs, training_outputs);

        // predicts the mean and variance of a single point
        let input = vec![1., 0.4];
        let mean = gp.predict(&input);
        let var = gp.predict_variance(&input);
        println!("prediction: {} ± {}", mean, var.sqrt());
    }
}
