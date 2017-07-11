#!/usr/bin/env bash

set -ex

NAME=gflags

# ---
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INSTALL_DIR=${DIR}/../install/${NAME}
mkdir -p ${INSTALL_DIR}
cd ${DIR}/../repos/${NAME}
# ---

mkdir -p build
cmake -H. -Bbuild \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DGFLAGS_NAMESPACE=google \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}

make -Cbuild -j12
make -Cbuild install
