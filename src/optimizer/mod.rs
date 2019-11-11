use crate::parameters::{kernel::Kernel, prior::Prior};
use crate::gaussian_process::GaussianProcess;

/*
  see [optimizing-gradient-descent](https://ruder.io/optimizing-gradient-descent/) for a good point on the various kind of gradient descent

  RMSPROP is very bad in the general case
  ADAM is bad in the simple case but very good in the hard case (!)

  see impact of first iter on ADAM type optimizer
  should we start at 0 or at a genuine value ?
*/

/// returns true if all gradients are below 1% of their corresponding parameter
fn gradients_too_small(parameters: &[f64], gradient: &[f64]) -> bool
{
   let fraction = 0.01; // 1%
   parameters.iter().zip(gradient.iter()).all(|(p, g)| fraction * p.abs() >= g.abs())
}

/// basic gradient descent
pub fn gradient_descent<KernelType: Kernel, PriorType: Prior>(gp: &mut GaussianProcess<KernelType,
                                                                                   PriorType>,
                                                              max_iter: usize)
{
   let epsilon = 0.1;

   let mut parameters = gp.get_parameters();
   println!("initial likelihood:{}\tinitial parameters:{:?}", gp.likelihood(), parameters);

   for i in 0..max_iter
   {
      let gradients = gp.gradient_marginal_likelihood();
      for (p, gradient) in parameters.iter_mut().zip(gradients.iter())
      {
         *p += epsilon * gradient;
      }
      gp.set_parameters(&parameters);

      println!("{}: likelihood {}\n- parameters {:?}\n- gradients  {:?}",
               i,
               gp.likelihood(),
               parameters,
               gradients);

      if gradients_too_small(&parameters, &gradients)
      {
         break;
      };
   }
}

/// ADAM optimizer
pub fn adam<KernelType: Kernel, PriorType: Prior>(gp: &mut GaussianProcess<KernelType, PriorType>,
                                                  max_iter: usize)
{
   // constant parameters
   let beta1 = 0.9;
   let beta2 = 0.999;
   let epsilon = 1e-8;
   let learning_rate = 0.1; // 0.001

   let mut parameters = gp.get_parameters();
   let mut mean_grad = vec![0.; parameters.len()];
   let mut var_grad = vec![0.; parameters.len()];
   let mut delta = vec![0.; parameters.len()];

   println!("initial likelihood:{}\tinitial parameters:{:?}", gp.likelihood(), parameters);

   for i in 1..=max_iter
   {
      let gradients = gp.gradient_marginal_likelihood();
      for p in 0..parameters.len()
      {
         mean_grad[p] = beta1 * mean_grad[p] + (1. - beta1) * gradients[p];
         var_grad[p] = beta2 * var_grad[p] + (1. - beta2) * gradients[p].powi(2);
         let bias_corrected_mean = mean_grad[p] / (1. - beta1.powi(i as i32));
         let bias_corrected_variance = var_grad[p] / (1. - beta2.powi(i as i32));
         delta[p] = bias_corrected_mean / (bias_corrected_variance.sqrt() + epsilon);
         parameters[p] -= learning_rate * delta[p]
      }
      gp.set_parameters(&parameters);

      println!("{}: likelihood {}\n- parameters {:?}\n- gradients  {:?}",
               i,
               gp.likelihood(),
               parameters,
               gradients);
      if gradients_too_small(&parameters, &delta)
      {
         break;
      };
   }
}

pub fn nadam<KernelType: Kernel, PriorType: Prior>(gp: &mut GaussianProcess<KernelType, PriorType>,
                                                   max_iter: usize)
{
   // constant parameters
   let beta1 = 0.9;
   let beta2 = 0.999;
   let epsilon = 1e-8;
   let learning_rate = 0.1;

   let mut parameters = gp.get_parameters();
   let mut mean_grad = vec![0.; parameters.len()];
   let mut var_grad = vec![0.; parameters.len()];
   let mut delta = vec![0.; parameters.len()];

   println!("initial likelihood:{}\tinitial parameters:{:?}", gp.likelihood(), parameters);

   for i in 1..=max_iter
   {
      let gradients = gp.gradient_marginal_likelihood();
      for p in 0..parameters.len()
      {
         mean_grad[p] = beta1 * mean_grad[p] + (1. - beta1) * gradients[p];
         var_grad[p] = beta2 * var_grad[p] + (1. - beta2) * gradients[p].powi(2);
         let bias_corrected_mean = mean_grad[p] / (1. - beta1.powi(i as i32));
         let bias_corrected_variance = var_grad[p] / (1. - beta2.powi(i as i32));
         delta[p] = (beta1 * bias_corrected_mean + (1. - beta1) * gradients[p] / (1. - beta1.powi(i as i32)))
                    / (bias_corrected_variance.sqrt() + epsilon);
         parameters[p] -= learning_rate * delta[p];
      }
      gp.set_parameters(&parameters);

      println!("{}: likelihood {}\n- parameters {:?}\n- gradients  {:?}",
               i,
               gp.likelihood(),
               parameters,
               gradients);
      if gradients_too_small(&parameters, &delta)
      {
         break;
      };
   }
}

pub fn amsgrad<KernelType: Kernel, PriorType: Prior>(gp: &mut GaussianProcess<KernelType, PriorType>,
                                                     max_iter: usize)
{
   // constant parameters
   let beta1 = 0.9;
   let beta2 = 0.999;
   let epsilon = 1e-8;
   let learning_rate = 0.1; // 0.001

   let mut parameters = gp.get_parameters();
   let mut mean_grad = vec![0.; parameters.len()];
   let mut var_grad = vec![0.; parameters.len()];
   let mut delta = vec![0.; parameters.len()];
   let mut max_var = 0.;

   println!("initial likelihood:{}\tinitial parameters:{:?}", gp.likelihood(), parameters);

   for i in 1..=max_iter
   {
      let gradients = gp.gradient_marginal_likelihood();
      for p in 0..parameters.len()
      {
         mean_grad[p] = beta1 * mean_grad[p] + (1. - beta1) * gradients[p];
         var_grad[p] = beta2 * var_grad[p] + (1. - beta2) * gradients[p].powi(2);
         max_var = if max_var > var_grad[p] { max_var } else { var_grad[p] };
         delta[p] = learning_rate * mean_grad[p] / (max_var.sqrt() + epsilon);
         parameters[p] -= delta[p];
      }
      gp.set_parameters(&parameters);

      println!("{}: likelihood {}\n- parameters {:?}\n- gradients  {:?}",
               i,
               gp.likelihood(),
               parameters,
               gradients);
      if gradients_too_small(&parameters, &delta)
      {
         break;
      };
   }
}
