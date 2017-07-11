#!/usr/bin/env bash

set -ex

NAME=opencv

# ---
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INSTALL_DIR=${DIR}/../install/${NAME}
mkdir -p ${INSTALL_DIR}
cd ${DIR}/../repos/${NAME}
# ---

mkdir -p build
cmake -H. -Bbuild \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DWITH_TBB=ON \
    -DWITH_EIGEN=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DWITH_OPENCL=OFF -DWITH_CUDA=OFF -DBUILD_opencv_gpu=OFF \
    -DBUILD_opencv_gpuarithm=OFF -DBUILD_opencv_gpubgsegm=OFF -DBUILD_opencv_gpucodec=OFF \
    -DBUILD_opencv_gpufeatures2d=OFF -DBUILD_opencv_gpufilters=OFF -DBUILD_opencv_gpuimgproc=OFF \
    -DBUILD_opencv_gpulegacy=OFF -DBUILD_opencv_gpuoptflow=OFF -DBUILD_opencv_gpustereo=OFF \
    -DBUILD_opencv_gpuwarping=OFF

make -Cbuild -j12
make -Cbuild install
