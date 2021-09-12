FROM nvidia/cudagl:11.1.1-devel-ubuntu20.04

RUN rm /etc/apt/sources.list.d/cuda.list

ENV DEBIAN_FRONTEND=noninteractive
ENV CODE_DIR=/home/jaxRBDL
#ENV SYS_DIR=/etc/apt

COPY ./scripts/ $CODE_DIR/scripts
COPY ./src/ $CODE_DIR/src
COPY ./test/ $CODE_DIR/test
COPY ./setup.py $CODE_DIR
COPY ./pyproject.toml $CODE_DIR
COPY ./CMakeLists.txt $CODE_DIR
COPY ./jaxrbdl_config.h.in $CODE_DIR
COPY ./README.md $CODE_DIR
#COPY ./sources.list $SYS_DIR

ENV PATH /usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda-11.1/compat:$LD_LIBRARY_PATH
ENV WITH_JAX_CUDA yes

RUN ln -s -f usr/lib/x86_64-linux-gnu/libcuda.so.460.73.01 /usr/local/cuda-11.1/compat/libcuda.so.1 && \
    echo export 'WITH_JAX_CUDA=yes' >> ~/.bashrc && \
    echo export 'PATH="/usr/local/cuda/bin:$PATH"'  >> ~/.bashrc && \
    echo export 'LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-11.1/compat:$LD_LIBRARY_PATH"' >> ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc"   && \
    apt-get update    && \
    apt-get install -y --quiet\
    vim\
    libxml2\
    wget\
    sudo \
    libx11-dev \
    tk \
    python3 \
    python3-tk \
    python3-matplotlib   && \
    wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /home/anaconda3 && \
    rm ~/anaconda.sh && \
    echo "export PATH=/home/anaconda3/bin:$PATH" >> ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc"  && \
    wget https://github.com/NVIDIA/cuda-samples/archive/refs/tags/v11.1.tar.gz  && \
    tar -zxvf v11.1.tar.gz  && \
    mkdir -p /usr/local/cuda-11.1/samples/common/inc  && \
    cp -r /cuda-samples-11.1/Common/* /usr/local/cuda-11.1/samples/common/inc


ENV PATH /home/anaconda3/bin:$PATH
RUN conda create -n jbdl_gpu python=3.8
ENV PATH /home/anaconda3/envs/jbdl_gpu/bin:$PATH
RUN /bin/bash -c "source activate jbdl_gpu" && \
    python -m pip -V && \
    pip install --upgrade pip && \
    pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html    


WORKDIR $CODE_DIR
RUN ls && \
    /bin/bash -c "source activate jbdl_gpu" && \
    python -m pip -V && \
    pip install -e .

