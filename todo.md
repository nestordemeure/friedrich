# TODO list

- how to know if a kernel is default defined or user defined ?
- add [benchmark](http://www.resibots.eu/limbo/release-2.0/reg_benchmarks.html)
- integrate all nalgebra fix once new version is released
- is it best to optimize one parameter at a time or all at once ? for fit ? for perf ? for memory ?
  (if one at a time is optimal for perf, it might be the best choice)
- some parameter, such as the amplitude of the gaussian kernel, can be analytically fitted but it is very kernel specific : can we do that ?
  [Fast methods for training Gaussian processes on large datasets](https://arxiv.org/pdf/1604.01250.pdf)
- introduce a macro (to have optional parameters with default value) instead of the builder pattern ? can it let us deal with knowing wich param is default ?

- log scale on noise could be dealt with directly in optimizer to make things cleaner...
