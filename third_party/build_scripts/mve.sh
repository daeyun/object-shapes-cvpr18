#!/usr/bin/env bash

set -ex

NAME=mve

# ---
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${DIR}/../repos/${NAME}
# ---


#https://github.com/simonfuhrmann/mve/issues/356

cd libs/
set +e
mv util/endian.h util/mve-endian.h
mv util/string.h util/mve-string.h

cd ..
grep -Rl 'util/endian.' . | xargs sed -i 's/util\/endian./util\/mve-endian./g'
grep -Rl 'util/string.' . | xargs sed -i 's/util\/string./util\/mve-string./g'
set -e

make -j12
