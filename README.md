# JBDL
A JAX-Based Body Dynamics Algorithm Library forRobotics.

# Citation

The technical report of this project:
Cheng Zhou, Lei Han, Yuzhu Mao. JBDL: A JAX-Based Body Dynamics Algorithm Library forRobotics. arXiv preprint arXiv:XXXX.XXXX, 2021. (correspondence to the first two authors)

The technical report will be released soon.

# Install
Go to home directory, and run the command:
```
pip install -e .
```
to install jbdl.

If you want to install jaxRBDL with both CPU and NVidia GPU support, you must first install CUDA and CuDNN and add environment variables:

```
echo export 'PATH="/usr/local/cuda/bin:$PATH"'  >> ~/.bashrc
echo export 'LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc
```
Before ```pip install -e .```, make sure the environment variable CUDA_PATH and option -DCMAKE_CUDA_COMPILER in ```setup.py``` are set to the CUDA Toolkit install directory:
```
CUDA_PATH = '/usr/local/cuda'
-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

The jaxlib version must correspond to the version of the existing CUDA installation. In ubuntu, you can check the cuda version by cheking
the folder under /usr/local/cuda.
* For CUDA 11.1, use cuda111. 
* For CUDA 11.0, use cuda110.
* Other CUDA versions has not been tested with jaxRBDL(cuosqp).

Take installation with CUDA 11.1 for example:
```
pip install --upgrade pip
pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

** A warning: Other versions of jax, jaxlib and python has not been tested. Some may lead to unpredictable errors. **

Then, set the environment variable ```WITH_JAX_CUDA=yes``` to enable CUDA support before installation:
```
export WITH_JAX_CUDA=yes
```

# Quick Examples
Go to demo directory under the scripts folder, and run
```
python test_experimental_math.py
```
or 
```
python test_half_max.py
```
to test the functionality of jbdl.

# License

Use MIT license (see LICENSE.md) except for third-party softwares. They are all open-source softwares and have their own licese types.
 
# Disclaimer
 
 This is not an officially supported Tencent product. The code and data in this repository are for research purpose only. No representation or warranty whatsoever, expressed or implied, is made as to its accuracy, reliability or completeness. We assume no liability and are not responsible for any misuse or damage caused by the code and data. Your use of the code and data are subject to applicable laws and your use of them is at your own risk.


