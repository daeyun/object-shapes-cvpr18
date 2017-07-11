#!/usr/bin/env bash

set -ex

NAME=tensorflow

# ---
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INSTALL_DIR=${DIR}/../install/${NAME}
mkdir -p ${INSTALL_DIR}
cd ${DIR}/../repos/${NAME}
# ---

## Define if not defined.
#export PYTHON_BIN_PATH=${PYTHON_BIN_PATH:-`which python`}
#export PYTHON_LIB_PATH=${PYTHON_LIB_PATH:-`python -c "import site; print(site.getsitepackages()[0])"`}
#
#export TF_NEED_CUDA=${TF_NEED_CUDA:-1}
#
#export CUDA_TOOLKIT_PATH=${CUDA_TOOLKIT_PATH:-/usr/local/cuda}
#export CUDNN_INSTALL_PATH=${CUDNN_INSTALL_PATH:-/usr/local/cuda}
#
#export TF_CUDA_VERSION=${TF_CUDA_VERSION:-8.0}
#
## Find cuDNN version.
#CUDNN_MAJOR=`cat ${CUDNN_INSTALL_PATH}/include/cudnn.h | sed -rn 's/.*CUDNN_MAJOR[\t ]+([0-9]+)/\1/p'`
#CUDNN_MINOR=`cat ${CUDNN_INSTALL_PATH}/include/cudnn.h | sed -rn 's/.*CUDNN_MINOR[\t ]+([0-9]+)/\1/p'`
#CUDNN_PATCHLEVEL=`cat ${CUDNN_INSTALL_PATH}/include/cudnn.h | sed -rn 's/.*CUDNN_PATCHLEVEL[\t ]+([0-9]+)/\1/p'`
#
#export TF_CUDNN_VERSION=${TF_CUDNN_VERSION:-"$CUDNN_MAJOR.$CUDNN_MINOR.$CUDNN_PATCHLEVEL"}
#export TF_CUDA_COMPUTE_CAPABILITIES=${TF_CUDA_COMPUTE_CAPABILITIES:-5.2,6.1}
#export GCC_HOST_COMPILER_PATH=${GCC_HOST_COMPILER_PATH:-/usr/bin/gcc}
#
#export TF_NEED_JEMALLOC=1
#export TF_NEED_GCP=0
#export TF_NEED_HDFS=0
#export TF_NEED_OPENCL=0
#export TF_ENABLE_XLA=0
#export CC_OPT_FLAGS="-march=native"
#export TF_NEED_MKL=1
#export TF_DOWNLOAD_MKL=1
#export TF_NEED_VERBS=0
#export TF_CUDA_CLANG=0
#
## ---
#
## Build
#
#./configure
#
#bazel build --config=opt --config=cuda //tensorflow:libtensorflow_cc.so
#
#bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
#
#
## Install
#
## https://github.com/tensorflow/tensorflow/issues/2412#issuecomment-300628873
mkdir -p ${INSTALL_DIR}/include/tf
mkdir -p ${INSTALL_DIR}/lib

cp --parents `find tensorflow -name \*.h` ../../install/tensorflow/include/tf/
cp -r --parents `find third_party \( -type f -o -type l \) -not -iname '*BUILD'` ../../install/tensorflow/include/tf/
cp -rf bazel-genfiles/ ${INSTALL_DIR}/include/tf/
cp -rf bazel-bin/tensorflow/libtensorflow_cc.so ${INSTALL_DIR}/lib/

# Output pip package.
#bazel-bin/tensorflow/tools/pip_package/build_pip_package ${INSTALL_DIR}/pkgs
