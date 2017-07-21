#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

set -ex

cd ${DIR}/third_party

nix-build -A env -o nix_env
