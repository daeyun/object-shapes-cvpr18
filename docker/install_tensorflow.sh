#!/usr/bin/env bash

set -ex

INSTALL_DIR=$HOME/usr/tensorflow
mkdir -p ${INSTALL_DIR}

# https://github.com/tensorflow/tensorflow/issues/2412#issuecomment-300628873
mkdir -p ${INSTALL_DIR}/include/tf
mkdir -p ${INSTALL_DIR}/lib

cp --parents `find tensorflow -name \*.h` ${INSTALL_DIR}/include/tf/
cp -r --parents `find third_party \( -type f -o -type l \) -not -iname '*BUILD'` ${INSTALL_DIR}/include/tf/
cp -rf bazel-genfiles/ ${INSTALL_DIR}/include/tf/
cp -rf bazel-bin/tensorflow/libtensorflow_cc.so ${INSTALL_DIR}/lib/

# Output pip package.
bazel-bin/tensorflow/tools/pip_package/build_pip_package ${INSTALL_DIR}/pkgs

pip3 install ${INSTALL_DIR}/pkgs/tensorflow*.whl
