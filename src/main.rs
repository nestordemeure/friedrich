mod kernel;
mod gaussian_process;
pub use gaussian_process::*;

fn main()
{
   let inputs: Vec<f32> = vec![];
   let gp = GaussianProcess::new(inputs);
   println!("Hello, world!");
}
