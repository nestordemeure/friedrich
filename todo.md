
- deal with possibility of, illegal, negative kernel parameter
  currently we avoid crash by taking abs when needed
  abs is not enough because it makes gradient wrong unless taken into account
  -> use abs and take it into account when computing gradient ?
- how to know if a kernel is default defined or user defined ?
- add benchmark http://www.resibots.eu/limbo/release-2.0/reg_benchmarks.html
- integrate all nalgebra fix once new version is released