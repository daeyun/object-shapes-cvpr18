#!/usr/bin/env bash

set -ex

NAME=gflags

# ---
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INSTALL_DIR=${DIR}/../install/${NAME}
mkdir -p ${INSTALL_DIR}
cd ${DIR}/../repos/${NAME}
# ---

# static
mkdir -p build
cmake -H. -Bbuild \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DGFLAGS_NAMESPACE=google \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DBUILD_SHARED_LIBS=OFF

make -Cbuild -j12
make -Cbuild install


# shared
mkdir -p build
cmake -H. -Bbuild \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DGFLAGS_NAMESPACE=google \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DBUILD_SHARED_LIBS=ON

make -Cbuild -j12
make -Cbuild install
