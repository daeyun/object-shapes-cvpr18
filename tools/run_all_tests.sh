#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJ_DIR=$DIR/..


MODE="debug"

while getopts ":r" opt; do
    case $opt in
        r)
            MODE="release"
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            ;;
    esac
done



set -ex

cd $PROJ_DIR

if [ $MODE = "debug" ]; then
    mkdir -p cmake-build-debug
    cmake -H. -Bcmake-build-debug -DCMAKE_BUILD_TYPE=Debug
    make runAllTests -Ccmake-build-debug -j 12
    cd $PROJ_DIR/cmake-build-debug/cpp/tests/
elif [ $MODE = "release" ]; then
    mkdir -p cmake-build-release
    cmake -H. -Bcmake-build-release -Dtest=OFF -DDEBUG=OFF -DCMAKE_BUILD_TYPE=Release
    make runAllTests -Ccmake-build-release -j 12
    cd $PROJ_DIR/cmake-build-release/cpp/tests/
fi


./runAllTests

echo "OK"
