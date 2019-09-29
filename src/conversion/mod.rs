mod input;
mod output;
mod slice;

pub use input::Input;
pub use output::Output;
pub use slice::{MatrixSlice, RowVectorSlice, VectorSlice, InputRef, OutputRef};

/*
   i need a conversion from value for the builder
   i need to go from ref to slice for the prediction
   => maybe i need one more trait for the prediciton, sliceable matrix seem like a good fit
   i should make a conversion module with input, output, inputref traits
   implement all of that properly
   and then go back to trained and see what is truly needed

   conversions should probably be done by the function that receive the type in order to make then generic
   without clutering implementation with conversions
*/


