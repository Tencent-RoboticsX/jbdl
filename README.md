# jbdl Version 0.80
## Installation.
Go to home directory, and run
```
pip install -e .
```
to install jbdl.

If you want to install jaxRBDL with both CPU and NVidia GPU support, you must first install CUDA and CuDNN, 
and make sure the environment variable CUDA_PATH and option -DCMAKE_CUDA_COMPILER in setup.py are set to the CUDA Toolkit install directory:
```
CUDA_PATH = '/usr/local/cuda'
-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

The jaxlib version must correspond to the version of the existing CUDA installation you want to use:
* For CUDA 11.1, use cuda111. 
* For CUDA 11.0, use cuda110.
* Other CUDA versions has not been tested with jaxRBDL(cuosqp).

Take installation with CUDA 11.1 for example:
```
wget https://storage.googleapis.com/jax-releases/cuda111/jaxlib-0.1.68+cuda111-cp36-none-manylinux2010_x86_64.whl
pip install jaxlib-0.1.68+cuda111-cp36-none-manylinux2010_x86_64.whl
pip install jax==0.2.16
```
** A warning: Other versions of jax, jaxlib and python has not been tested. Some may lead to unpredictable errors. **

Then, set the environment variable ```LCP_JAX_CUDA=yes``` to enable CUDA  CUDA support before installation:
```
vim ~/.bashrc
export LCP_JAX_CUDA="yes"
source ~/.bashrc
```

## Examples.
Go to demo directory under the scripts folder, and run
```
python test_half_max_v5.py
```
or 
```
python test_whole_max_v2.py
```
to test the functionality of jbdl.
 
