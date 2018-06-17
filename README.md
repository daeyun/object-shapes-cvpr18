This repository contains the source code for the following paper:

**Pixels, voxels, and views: A study of shape representations for single view 3D object shape prediction**  
Daeyun Shin, Charless Fowlkes, Derek Hoiem  
In *CVPR*, 2018  &nbsp; [\[pdf\]](https://www.ics.uci.edu/~daeyuns/pixels-voxels-views/CVPR18_Shin.pdf) &nbsp;  [\[supplementary material\]](https://www.ics.uci.edu/~daeyuns/pixels-voxels-views/CVPR18_Shin_Supp.pdf) &nbsp;  [\[poster\]](https://www.ics.uci.edu/~daeyuns/pixels-voxels-views/CVPR18_Shin_Poster.pdf)

### Project page

https://www.ics.uci.edu/~daeyuns/pixels-voxels-views/


### Installation

This code base is not fully deployable yet (see todo list below). But in the meantime, I can point to where you can find relevant code:

- Pytorch model: [encoderdecoder.py#L32](python/mvshape/models/encoderdecoder.py#L32)
- Dataset generation: [generate_dataset.cc](cpp/apps/generate_dataset.cc)
- Evaluation: [meshdist.cc](cpp/apps/meshdist.cc), [voxel_iou.cc](cpp/apps/voxel_iou.cc)
- SQLite database schema (data generation code takes this database as input): [db_model.py](python/mvshape/db_model.py)

#### Todo

- [ ] Change hard-coded paths to relative paths.
- [ ] Add an environment.yml file for the Anaconda virtual environment.
- [ ] Provide a download scripts to download the datasets.
- [ ] Upload pre-compiled C++ binaries.
- [ ] Upload sqlite database needed for dataset generation.
- [ ] Upload pre-trained weights for the Pytorch model.

In the meantime, if you need to get any specific part working for research, emailing me is probably the fastest. Contact info can be found [here](http://research.dshin.org/)


#### C++

Clone the repository with the --recurse-submodules option to download third party dependencies.

```
git clone --recurse-submodules git@github.com:daeyun/scene3d.git
```

Then run the build scripts in [third_party/build_scripts](third_party/build_scripts) to compile the third party libraries if you don't have these dependencies locally installled.

Finally, run this script to build the binaries:

```
./tools/build_all.sh
```

#### Python

Anaconda environment file coming soon.
