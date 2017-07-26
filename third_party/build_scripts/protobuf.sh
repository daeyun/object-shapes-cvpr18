#!/usr/bin/env bash

set -ex

NAME=protobuf

# ---
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INSTALL_DIR=${DIR}/../install/${NAME}
mkdir -p ${INSTALL_DIR}
cd ${DIR}/../repos/${NAME}
# ---

./autogen.sh
./configure --prefix ${INSTALL_DIR}

make -j12
make install
