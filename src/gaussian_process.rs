// https://github.com/AtheMathmo/rusty-machine/blob/master/src/learning/gp.rs
/*
   a difficulty is that gaussian process can be applied to any inputs on which a distance metric is computed

   further more their output are not restricted to a single number, you can use anything including a vector of complex
   as long as you can perform a weighted sum on the output

   now, how do we express this flexibility so that the user can benefit from it
   while not building a system that is too complex

   there is also the problem of the bandwidth computation
   do we keep the formula to ourselves or do so we let the user define it or...
   a constant bandwith and a bandwidth recomputed at each iteration will lead to very different profiles

   set_prior
   set_kernel
   set_noise
   set_bandwidth ? (make it a kernel parameter ?)

   training_input // things that implement distance or nalgebra matrix (suffice to implement the distance trait on it)
   training_output // nalgebra matrix (which includes vector making it easy for the user)

   add_point(input, output)
   mean(input) -> output
   mean_cov(input) -> output, covarianceMatrix
   mean_weightedVar(input) -> output, variancesVector
   sample(input) -> output
   fit_bandwith() ?
   start with implem on matrix -> matrix
   and later find a trait to implment by default on matrix and that
   
   there are two stages to gaussian process :
   - raw data
   - fitted kernel
   (the fit being the selection of the parameter
   + the computation of the kernel matrix
   + the computation of its cholesky decomposition and so on)
   
   let gp: GaussianProcess = GaussianProcess::new(inputs, outputs).fit()
   one can add data (which will be used to make further predictions)
   and fit (which will be used to choose parameters)
   new produces a gaussianprocessbuilder that requires a call to fit in order to be used ?
*/

/// gaussian process
pub struct GaussianProcess<InputType>
{
   training_inputs: Vec<InputType>
}

//---------------------------------------------------------------------------------------
// Constructor

pub trait New<InputCollection, InputType>
{
   fn new(training_inputs: InputCollection) -> GaussianProcess<InputType>;
}

impl<InputType> New<Vec<InputType>, InputType> for GaussianProcess<InputType>
{
   /// constructor overloaded to deal with general inputs on which a distance has been defined
   fn new(training_inputs: Vec<InputType>) -> GaussianProcess<InputType>
   {
      GaussianProcess { training_inputs }
   }
}

impl New<&[f32], f32> for GaussianProcess<f32>
{
   /// constructor overloaded to deal with matrix input
   fn new(training_inputs: &[f32]) -> GaussianProcess<f32>
   {
      let training_inputs = training_inputs.to_vec();
      GaussianProcess { training_inputs }
   }
}
