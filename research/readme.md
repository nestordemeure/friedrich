# Ideas

## Kernel regression

Kernel regression is close to gaussian process regression but does not require any complex computation.

It does not fit the data as well but it is `o(n)`.

I believe I could compute a variance and add a prior for the [Nadaraya–Watson algorithm](https://en.wikipedia.org/wiki/Kernel_regression#Nadaraya–Watson_kernel_regression) making it a suitable replacement for Gaussian process when we are after its properties and not its perfect fit.

## K nearest neighbours

We could compute the output of a point using only its k nearest neigbours (which could be computed using [Rtree](https://docs.rs/spade/1.8.0/spade/rtree/struct.RTree.html)).
Thus the algorithm would become `o(k^3 * log(n))` instead of `o(n^3)`.

A downside is that with this formulation, we cannot reuse a Cholesky decomposition from a point to another.
