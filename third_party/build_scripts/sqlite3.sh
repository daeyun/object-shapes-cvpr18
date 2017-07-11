#!/usr/bin/env bash

set -ex

NAME=sqlite3

# ---
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INSTALL_DIR=${DIR}/../install/${NAME}
mkdir -p ${INSTALL_DIR}
# ---

cd ${INSTALL_DIR}

wget -N https://sqlite.org/2017/sqlite-autoconf-3190300.tar.gz

tar xvzf sqlite-autoconf-3190300.tar.gz

cd sqlite-autoconf-3190300

./configure --prefix ${INSTALL_DIR}

make -j12
make install
