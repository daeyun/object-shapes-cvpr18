from mvshape import io_utils
import glob
import collections
import numpy as np
import os.path as path
import blosc
from mvshape.proto import dataset_pb2


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

        tensors[name] = arr

    indices = tensors['index'].ravel().copy()
    argsort = np.argsort(indices)

    ret = {k: v[argsort] for k, v in tensors.items()}

    return ret


def read_split_metadata(split_file, subset_tags=None):
    """

    :param split_file:
    :param subset_tags: For example, ['NOVELCLASS', 'NOVELMODEL', 'NOVELVIEW']
    :return:
    """
    with open(split_file, mode='rb') as f:
        compressed = f.read()
    decompressed = blosc.decompress(compressed)
    examples = dataset_pb2.Examples()
    examples.ParseFromString(decompressed)

    if subset_tags is not None:
        assert isinstance(subset_tags, (list, tuple))
        experiment_names = subset_tags

        ret = collections.defaultdict(list)

        for example in examples.examples:
            tags = tags_from_example(example)
            for name in experiment_names:
                if name in tags:
                    ret[name].append(example)
    else:
        ret = list(examples.examples)

    return ret


def tags_from_example(pb_example: dataset_pb2.Example):
    tag_names = []
    for tag_id in pb_example.tags:
        tag_names.append(dataset_pb2.Tag.Name(tag_id))
    return tag_names
