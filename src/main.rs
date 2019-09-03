mod kernel;
mod gaussian_process;

fn main()
{
   // test pattern
   let training_inputs = vec![1, 2, 3];
   for (sample_index, sample) in training_inputs.iter().enumerate()
   {
      for sample2 in training_inputs.iter().skip(sample_index + 1)
      {
         println!("{} - {}", sample, sample2);
      }
   }
}
