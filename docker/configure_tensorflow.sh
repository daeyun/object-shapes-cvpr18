#!/usr/bin/env bash

set -ex

NAME=tensorflow

# ---

# Define if not defined.
export PYTHON_BIN_PATH=${PYTHON_BIN_PATH:-`which python3`}
export PYTHON_LIB_PATH=${PYTHON_LIB_PATH:-`python3 -c "import site; print(site.getsitepackages()[0])"`}

export TF_NEED_CUDA=${TF_NEED_CUDA:-1}

export CUDA_TOOLKIT_PATH=${CUDA_TOOLKIT_PATH:-/usr/local/cuda}
export CUDNN_INSTALL_PATH=${CUDNN_INSTALL_PATH:-/usr}

export TF_CUDA_VERSION=${TF_CUDA_VERSION:-8.0}

# Find cuDNN version.
CUDNN_MAJOR=`cat ${CUDNN_INSTALL_PATH}/include/cudnn.h | sed -rn 's/.*CUDNN_MAJOR[\t ]+([0-9]+)/\1/p'`
CUDNN_MINOR=`cat ${CUDNN_INSTALL_PATH}/include/cudnn.h | sed -rn 's/.*CUDNN_MINOR[\t ]+([0-9]+)/\1/p'`
CUDNN_PATCHLEVEL=`cat ${CUDNN_INSTALL_PATH}/include/cudnn.h | sed -rn 's/.*CUDNN_PATCHLEVEL[\t ]+([0-9]+)/\1/p'`

export TF_CUDNN_VERSION=${TF_CUDNN_VERSION:-"$CUDNN_MAJOR.$CUDNN_MINOR.$CUDNN_PATCHLEVEL"}
export TF_CUDA_COMPUTE_CAPABILITIES=${TF_CUDA_COMPUTE_CAPABILITIES:-5.2,6.1}
export GCC_HOST_COMPILER_PATH=${GCC_HOST_COMPILER_PATH:-/usr/bin/gcc}

export TF_NEED_JEMALLOC=1
export TF_NEED_GCP=0
export TF_NEED_HDFS=0
export TF_NEED_OPENCL=0
export TF_ENABLE_XLA=0
export CC_OPT_FLAGS="-march=native"
export TF_NEED_MKL=1
export TF_DOWNLOAD_MKL=1
export TF_NEED_VERBS=0
export TF_CUDA_CLANG=0

# ---

./configure
