from mvshape import io_utils
import glob
import numpy as np
import os.path as path


def read_tensors(basedir):
    # basedir = "/data/mvshape/out/tf_out/depth_6_views/object_centered/tensors/0040_000011400/NOVELVIEW/"
    tensor_names = [
        'out_silhouette',
        'out_depth',
        'placeholder_target_depth',
        'placeholder_in_depth',
        'index',
    ]

    tensors = {}

    for name in tensor_names:
        files = sorted(glob.glob('{}/*.bin'.format(path.join(basedir, name))))
        if name == 'index':
            dtype = np.int32
        else:
            dtype = np.float32
        arr = np.concatenate([io_utils.read_array_compressed(file, dtype) for file in files], axis=0)
        print(arr.shape)

        tensors[name] = arr

    indices = tensors['index'].ravel().copy()
    argsort = np.argsort(indices)

    ret = {k: v[argsort] for k, v in tensors.items()}

    return ret


if __name__ == '__main__':
    main()
