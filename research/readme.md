# Ideas

## Inputs

The algorithm can be used with **any** datatype as long as there is a distance function defined between points.

It can also be use with any outputs (including multidimentional outputs and complex numbers) as long as one can compute a weighted sum over the outputs.

## Kernel

Silverman's rule of thumb (using th emean distance between the points instead of the std) proved to be a very good default value for the bandwidth in my tests.

## Weighted variance

The covariance (and variance) provided by the algorithm is **fully** independant of the value of the outputs.
It is only a function of the distance to the known points and the kernel used.
Which means that we should probably scale it by the std of the outputs in order for it to be properly scaled.

A variance that would be truly function of the points would be the [weighted variance](http://re-design.dimiter.eu/?p=290) of the training points weighted by the weights produced by the algorithm.
This quantity would be close to 0 where things are well known and large where they are not.
The one problem is that it tends to be constant and equal to thelast value observed at infinity.

## Kernel regression

Kernel regression is close to gaussian process regression but does not require any complex computation.

It does not fit the data as well but it is `o(n)`.

I believe I could compute a weighted variance and add a prior for the [Nadaraya–Watson algorithm](https://en.wikipedia.org/wiki/Kernel_regression#Nadaraya–Watson_kernel_regression) making it a suitable replacement for Gaussian process when we are after its properties and not its perfect fit.

## K nearest neighbours

We could compute the output of a point using only its k nearest neigbours.
Thus the algorithm would become `o(k^3 * log(n))` instead of `o(n^3)`.

A downside is that with this formulation, we cannot reuse a Cholesky decomposition from an input point to another.
Overall this make computing several points at the same time much more complex

For the nearest neighbours search, we could use one of :

- [hnsw](https://crates.io/crates/hnsw)
- [kdtree-rs](https://github.com/mrhooray/kdtree-rs)
- [vpsearch](https://crates.io/crates/vpsearch)
- [Rtree](https://docs.rs/spade/1.8.0/spade/rtree/struct.RTree.html)
- [ball-tree](https://crates.io/crates/ball-tree)

## Hierarchical gaussian process

Inspired by [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/pdf/1603.09320.pdf).

One could use a hierarchical representation (build using the hnsw trick of random depth in order to deal with adding new points) where each layer is more precise / local than the previous one and uses the previous layer as a prior.
This formulation could use precomputed choleski decomposition, be fast and localy accurate.

The one difficulty is the frontier between local hierarchies...
