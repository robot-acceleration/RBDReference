# RBDReference

A Python reference implementation of rigid body dynamics algorithms.

This package is designed to enable rapid prototyping and testing of new algorithms and algorithmic optimizations. If your favorite rigid body dynamics algorithm is not yet implemented please submit a PR with the implementation. We'll then try to get a GPU, FPGA, and/or accelerator implementation designed as soon as possible.

## Usage and API:
This package relies on an already parsed ```robot``` object from our [URDFParser](https://github.com/robot-acceleration/URDFParser) package.
```python
RBDReference = RBDReference(robot)
outputs = RBDReference.ALGORITHM(inputs)
```

Currently implemented algorithms include the:
+ Recursive Newton Euler Algorithm (RNEA): ```(c,v,a,f) = rbdReference.rnea(q, qd, qdd = None, GRAVITY = -9.81)```
+ The Gradient of the RNEA: ```dc_du = rnea_grad(q, qd, qdd = None, GRAVITY = -9.81)``` where ```dc_du = np.hstack((dc_dq,dc_dqd))```
+ The Direct Inverse of the Mass Matrix Algorithm: ```Minv = rbdReference.minv(q, output_dense = True)```

We also include functions that break these algorithms down into there different passes and by their output types (dq vs dqd) to enable easier testing of downstream GPU, FPGA, and accelerator implementations.

## Instalation Instructions::
The only external dependency is ```numpy``` which can be automatically installed by running:
```shell
pip3 install -r requirements.txt
```
This package also depends on our [URDFParser](https://github.com/robot-acceleration/URDFParser) package.
