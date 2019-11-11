use crate::parameters::{kernel::Kernel, prior::Prior};
use crate::gaussian_process::GaussianProcess;

/// see [optimizing-gradient-descent](https://ruder.io/optimizing-gradient-descent/) for a good point on the various kind of gradient descent
/// stops when all gradient are below 1% of their parameters

/// returns true if all gradients are below 1% of their corresponding parameter
fn gradients_too_small(parameters: &[f64], gradient: &[f64]) -> bool
{
   let fraction = 0.01; // 1%
   parameters.iter().zip(gradient.iter()).all(|(p, g)| fraction * p.abs() >= g.abs())
}

/// basic gradient descent
/// very good on basic case, very bad on degenerate case
pub fn gradient_descent<KernelType: Kernel, PriorType: Prior>(gp: &mut GaussianProcess<KernelType,
                                                                                   PriorType>,
                                                              max_iter: usize,
                                                              epsilon: f64)
{
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
/// bad on basic case, very good on difficult case
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
         parameters[p] -= (learning_rate * bias_corrected_mean) / (bias_corrected_variance.sqrt() + epsilon)
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

/// RMSPROP
/// similar to ADAM in perfs
pub fn rmsprop<KernelType: Kernel, PriorType: Prior>(gp: &mut GaussianProcess<KernelType, PriorType>,
                                                     max_iter: usize)
{
   // constant parameters
   let beta2 = 0.9;
   let epsilon = 1e-8;
   let learning_rate = 0.1; // 0.001

   let mut parameters = gp.get_parameters();
   let mut var_grad = vec![0.; parameters.len()];

   println!("initial likelihood:{}\tinitial parameters:{:?}", gp.likelihood(), parameters);

   for i in 1..=max_iter
   {
      let gradients = gp.gradient_marginal_likelihood();
      for p in 0..parameters.len()
      {
         var_grad[p] = beta2 * var_grad[p] + (1. - beta2) * gradients[p].powi(2);
         parameters[p] -= (learning_rate * gradients[p]) / (var_grad[p] + epsilon).sqrt()
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
