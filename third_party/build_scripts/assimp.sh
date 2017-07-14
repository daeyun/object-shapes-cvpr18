#!/usr/bin/env bash

set -ex

NAME=assimp

# ---
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INSTALL_DIR=${DIR}/../install/${NAME}
mkdir -p ${INSTALL_DIR}
cd ${DIR}/../repos/${NAME}
# ---

# Static library

mkdir -p build
cmake -H. -Bbuild \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DBUILD_SHARED_LIBS=OFF

make -Cbuild -j12
make -Cbuild install


# Shared library

mkdir -p build_shared
cmake -H. -Bbuild_shared \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DBUILD_SHARED_LIBS=ON

make -Cbuild_shared -j12
make -Cbuild_shared install
