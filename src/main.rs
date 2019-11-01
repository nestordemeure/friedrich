#![allow(dead_code)]

mod parameters;
mod gaussian_process;
mod algebra;
mod conversion;

use crate::gaussian_process::GaussianProcess;
use crate::parameters::prior::*;
use crate::parameters::kernel::*;

fn main()
{
   // training data
   let training_inputs: Vec<_> = [0.8, 1.2, 3.8, 4.2].iter().map(|&x| vec![x]).collect();
   let training_outputs = vec![3.0, 4.0, -2.0, -2.0];

   let input_dimension = 1;
   let output_noise = 0.1;
   let exponential_kernel = Exponential::default();
   let linear_prior = LinearPrior::default(input_dimension);

   // builds a model
   //let mut gp = GaussianProcess::default(training_inputs, training_outputs);
   let mut gp = GaussianProcess::builder(training_inputs, training_outputs).set_noise(output_noise)
                                                                           .set_kernel(exponential_kernel)
                                                                           .fit_kernel()
                                                                           .set_prior(linear_prior)
                                                                           .fit_prior()
                                                                           .train();
   gp.fit_parameters(true, true);

   // make a prediction on new data
   let inputs: Vec<_> = vec![1.0, 2.0, 3.0, 4.2, 7.].iter().map(|&x| vec![x]).collect();
   let outputs = gp.predict(&inputs);
   println!("prediction: {:?}", outputs);
   let var = gp.predict_variance(&inputs);
   println!("standard deviation: {:?}", var);

   // updates the model
   let additional_inputs: Vec<_> = vec![0., 1., 2., 5.].iter().map(|&x| vec![x]).collect();
   let additional_outputs = vec![2.0, 3.0, -1.0, -2.0];
   let fit_prior = true;
   let fit_kernel = true;
   gp.add_samples_fit(&additional_inputs, &additional_outputs, fit_prior, fit_kernel);

   // renew prediction
   let outputs = gp.predict(&inputs);
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
