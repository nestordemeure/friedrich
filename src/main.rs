#![allow(dead_code)]

mod parameters;
mod gaussian_process;
mod matrix;

use nalgebra::{DMatrix};
use crate::gaussian_process::GaussianProcess;

fn main()
{
   // training data
   let training_inputs = DMatrix::from_column_slice(4, 1, &[0.8, 1.2, 3.8, 4.2]);
   let training_outputs = DMatrix::from_column_slice(4, 1, &[3.0, 4.0, -2.0, -2.0]);

   // builds a model
   let gp = GaussianProcess::default(training_inputs, training_outputs);

   // make a prediction on new data
   let inputs = DMatrix::from_column_slice(4, 1, &[1.0, 2.0, 3.0, 4.2]);
   let outputs = gp.predict_mean(inputs);
   println!("output: {}", outputs);
}
