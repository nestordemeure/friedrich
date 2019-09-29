#![allow(dead_code)]

mod multivariate_normal;
mod parameters;
mod gaussian_process;
mod algebra;
mod conversion;

use crate::gaussian_process::GaussianProcess;

fn main()
{
   // training data
   let training_inputs = vec![0.8, 1.2, 3.8, 4.2];
   let training_outputs = vec![3.0, 4.0, -2.0, -2.0];

   // builds a model
   let mut gp = GaussianProcess::default(training_inputs, training_outputs);

   // make a prediction on new data
   let inputs = vec![1.0, 2.0, 3.0, 4.2, 7.];
   let outputs = gp.predict_mean(&inputs);
   println!("prediction: {:?}", outputs);
   let var = gp.predict_variance(&inputs);
   println!("standard deviation: {:?}", var);

   // updates the model
   let additional_inputs = vec![0., 1., 2., 5.];
   let additional_outputs = vec![2.0, 3.0, -1.0, -2.0];
   let fit_prior = true;
   let fit_kernel = true;
   gp.add_samples_fit(&additional_inputs, &additional_outputs, fit_prior, fit_kernel);

   // renew prediction
   let outputs = gp.predict_mean(&inputs);
   println!("prediction 2: {:?}", outputs);
   let var = gp.predict_variance(&inputs);
   println!("standard deviation 2: {:?}", var);

   // sample the gaussian process on new data
   let sampler = gp.sample_at(&inputs);
   let mut rng = rand::thread_rng();
   for i in 1..=5
   {
      println!("sample {}: {:?}", i, sampler.sample(&mut rng));
   }
}
