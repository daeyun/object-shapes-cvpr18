```bash
bazel run --config=opt //cpp/apps:generate_dataset -- --num_processes=10 --resources_dir=$HOME/git/mvshape/resources
```

```bash
bazel run -c opt --config=opt //cpp/apps:train -- --tf_model=/data/mvshape/out/tf_models/depth_6_views/ --batch_size=100 --run_id=temp14  --resources_dir=$HOME/git/mvshape/resources
```

```bash
bazel test -c opt //cpp/tests:all --nocache_test_results
```