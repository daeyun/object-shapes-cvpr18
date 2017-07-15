Run all tests and build all binaries.

```bash
bazel test -c opt //cpp/tests:all --nocache_test_results && bazel build --config=release
```

Generate dataset

```bash
./bazel-bin/cpp/apps/generate_dataset --num_processes=10 --resources_dir=$HOME/git/mvshape/resources
```

Generate metadata

```bash
./bazel-bin/cpp/apps/generate_metadata --resources_dir=$HOME/git/mvshape/resources
```

Train

```bash
./bazel-bin/cpp/apps/train --tf_model=/data/mvshape/out/tf_models/depth_6_views/ --batch_size=100 --run_id=temp14  --resources_dir=$HOME/git/mvshape/resources
```

