use crate::parameters::{kernel::Kernel, prior::Prior};
use crate::gaussian_process::GaussianProcess;

pub fn adam<KernelType: Kernel, PriorType: Prior>(gp: &mut GaussianProcess<KernelType, PriorType>,
                                                  max_iter: usize)
{
   // use the ADAM gradient descent algorithm
   // see [optimizing-gradient-descent](https://ruder.io/optimizing-gradient-descent/)
   // for a good point on current gradient descent algorithms

   // constant parameters
   let beta1 = 0.9;
   let beta2 = 0.999;
   let epsilon = 1e-8;
   let learning_rate = 0.1;
   let convergence_fraction = 0.01; // if delta goes below this fraction of all parameters then we stop

   let mut parameters = gp.get_parameters();
   let mut mean_grad = vec![0.; parameters.len()];
   let mut var_grad = vec![0.; parameters.len()];
   for i in 1..=max_iter
   {
      let gradients = gp.gradient_marginal_likelihood();

      let mut continue_search = false;
      for p in 0..parameters.len()
      {
         mean_grad[p] = beta1 * mean_grad[p] + (1. - beta1) * gradients[p];
         var_grad[p] = beta2 * var_grad[p] + (1. - beta2) * gradients[p].powi(2);
         let bias_corrected_mean = mean_grad[p] / (1. - beta1.powi(i as i32));
         let bias_corrected_variance = var_grad[p] / (1. - beta2.powi(i as i32));
         let delta = learning_rate * bias_corrected_mean / (bias_corrected_variance.sqrt() + epsilon);
         continue_search |= delta.abs() > parameters[p].abs() * convergence_fraction;
         parameters[p] += delta;
      }

      gp.set_parameters(&parameters);
      if !continue_search
      {
         println!("finished in {} iter, likehihood {} for {:?}", i, gp.likelihood(), parameters);
         break;
      };
   }
}
