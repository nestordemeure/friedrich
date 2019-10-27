#![allow(dead_code)]

mod parameters;
mod gaussian_process;
mod algebra;
mod conversion;

use crate::gaussian_process::GaussianProcess;
use crate::parameters::kernel::Kernel;
use crate::parameters::prior::Prior;

// testing deref
fn print_noise<K: Kernel, P: Prior>(gp: &GaussianProcess<K, P>)
{
   println!("noise: {}", gp.noise)
}

fn main()
{
   // training data
   let training_inputs: Vec<_> = [0.8, 1.2, 3.8, 4.2].iter().map(|&x| vec![x]).collect();
   let training_outputs = vec![3.0, 4.0, -2.0, -2.0];

   // builds a model
   let mut gp = GaussianProcess::default(training_inputs, training_outputs);
   print_noise(&gp);
   /*let mut gp = GaussianProcess::new(&training_inputs, &training_outputs).set_noise(0.1f64)
   .fit_kernel()
   .fit_prior()
   .train();*/
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
