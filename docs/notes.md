Dependencies on Ubuntu 16.04

```
sudo apt install libgoogle-glog-dev libboost-all-dev libeigen3-dev libassimp-dev libgtest-dev libsqlite3-dev libegl1-mesa-dev libgles2-mesa-dev libpng12-dev libjpeg8-dev libtiff5-dev libglm-dev libprotobuf-dev
```




```
./cmake-build-release/cpp/apps/generate_dataset --num_processes=10 --data_dir=/data/mvshape  --out_dir=/data/mvshape/out
./cmake-build-release/cpp/apps/generate_metadata --data_dir=/data/mvshape --out_dir=/data/mvshape/out
```


Updating the subset in resources dir. First make sure `resources/data/database/shrec12.sqlite` is up-to-date.
```
./cmake-build-release/cpp/apps/generate_metadata --data_dir=./resources/data --out_dir=./resources/data
```

Currently you have to manually generate the views in the subset db.


Generating python protobuf code.
```
protoc --proto_path ~/git/mvshape/proto --python_out ./python/mvshape/proto ~/git/mvshape/proto/dataset.proto
```


