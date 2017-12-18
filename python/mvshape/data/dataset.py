import random
from os import path

import numpy as np
import cbor

from mvshape import eval_out_reader
from mvshape.data import parallel_reader

base = path.realpath(path.expanduser('~/data/mvshape'))

shrec12_target_depth_offset = -5.5
shapenetcore_target_depth_offset = -2

shrec12_train_category_ids = np.array(
    sorted([1, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 26, 27, 29, 30, 32, 34,
            35, 36, 38, 42, 43, 44, 46, 48, 49, 50, 51, 53, 54, 55, 56, 58, 59]))


def make_chunks(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def get_field_filenames(example_pb_list, fieldname):
    filenames = []
    for item in example_pb_list:
        name = getattr(item, fieldname).filename
        filename = path.join(base, name[1:])
        filenames.append(filename.encode('utf-8'))
    return filenames


def get_field_filenames2(example_dict_list, fieldname):
    filenames = []
    for item in example_dict_list:
        name = item[fieldname]['filename']
        filename = path.join(base, name[1:])
        filenames.append(filename.encode('utf-8'))
    return filenames


class ExampleLoader(object):
    def __init__(self, metadata_filename,
                 tensors_to_read=('single_depth', 'multiview_depth'),
                 batch_size=100,
                 subset_tags=None,
                 shuffle=False):
        """
        :param metadata_filename:
            e.g. '/data/mvshape/out/splits/shrec12_examples_vpo/train.bin'
        :param subset_tags:
            For example, ['NOVELCLASS', 'NOVELMODEL', 'NOVELVIEW']
            None for training.
        """
        self.examples_filename = metadata_filename
        self.tensors_to_read = tensors_to_read
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.examples = eval_out_reader.read_split_metadata(path.realpath(self.examples_filename), subset_tags=subset_tags)

        if subset_tags is not None:
            new_examples = []
            for item in subset_tags:
                new_examples.extend(self.examples[item])
            self.examples = new_examples

        self.current_batch_index = 0
        self.batched_examples = None
        self.reset()

        # Defines all available field names, sizes, and datatypes.
        self.available_fields = {
            'single_depth': ((None, 1, 128, 128), np.float32),
            'multiview_depth': ((None, 6, 128, 128), np.float32),
        }

        for fieldname in self.tensors_to_read:
            assert fieldname in self.available_fields, 'invalid field name {}'.format(fieldname)

    def reset(self):
        if self.shuffle:
            random.shuffle(self.examples)
        self.batched_examples = list(make_chunks(self.examples, n=self.batch_size))
        self.current_batch_index = 0

    def next(self):
        """
        :return: (examples, tensors)
        """
        if self.current_batch_index >= len(self.batched_examples):
            return None

        # List of Example protobufs.
        batch = self.batched_examples[self.current_batch_index]
        self.current_batch_index += 1

        tensors = {}
        for fieldname in self.tensors_to_read:
            shape, dtype = self.available_fields[fieldname]
            filenames = get_field_filenames(batch, fieldname)
            assert shape[0] is None
            shape = (len(batch),) + shape[1:]
            tensors[fieldname] = parallel_reader.read_batch(filenames, shape=shape, dtype=dtype)

        return batch, tensors


class ExampleLoader2(object):
    def __init__(self, metadata_filename,
                 tensors_to_read=('single_depth', 'multiview_depth'),
                 batch_size=100,
                 subset_tags=None,
                 shuffle=False, split_name=None):
        """
        :param metadata_filename:
            e.g. '/data/mvshape/out/splits/shrec12_examples_vpo/train.bin'
        :param subset_tags:
            For example, ['NOVELCLASS', 'NOVELMODEL', 'NOVELVIEW']
            None for training.
        :param split_name:
            Only used for shrec12 currently.
        """
        self.examples_filename = metadata_filename
        self.tensors_to_read = tensors_to_read
        self.batch_size = batch_size
        self.shuffle = shuffle

        # self.examples = eval_out_reader.read_split_metadata(path.realpath(self.examples_filename), subset_tags=subset_tags)
        assert '.cbor' in metadata_filename, metadata_filename
        with open(metadata_filename, 'rb') as f:
            self.all_data = cbor.load(f)

        # TODO: handle subset_tags
        if subset_tags is not None:
            raise NotImplementedError()

        # TODO: making sure things aren't broken by the new logic. they keys should be one of TRAIN TEST VALIDATION and extra two input_resolution and target_resolution.
        if 'shapenetcore' in metadata_filename or 'pascal3d' in metadata_filename:
            assert split_name is None  # not needed for shapenetcore
            assert len(self.all_data.keys()) == 3

            if 'TRAIN' in self.all_data:
                data_key = 'TRAIN'
                assert 'TEST' not in self.all_data
                assert 'VALIDATION' not in self.all_data
                data_key = 'TRAIN'
            elif 'TEST' in self.all_data:
                data_key = 'TEST'
                assert 'VALIDATION' not in self.all_data
                assert 'TRAIN' not in self.all_data
                data_key = 'TEST'
            elif 'VALIDATION' in self.all_data:
                data_key = 'VALIDATION'
                assert 'TEST' not in self.all_data
                assert 'TRAIN' not in self.all_data
                data_key = 'VALIDATION'
            else:
                raise NotImplementedError()
            # TODO: hack
            self.data_being_used = self.all_data[data_key]

            self.available_fields = {
                'input_depth': ((None, 1, 128, 128), np.float32),
                'input_rgb': ((None, 3, 128, 128), np.uint8),  # png
                'target_depth': ((None, 6, 128, 128), np.float32),
                'target_voxels': ((None, 1, 32, 32, 32), np.uint8),
            }
        elif 'shrec12' in metadata_filename:
            if split_name is None:
                self.data_being_used = self.all_data['TRAIN'] + self.all_data['VALIDATION']
            else:
                self.data_being_used = self.all_data[split_name]

            self.available_fields = {
                'input_depth': ((None, 1, 128, 128), np.float32),
                'input_rgb': ((None, 3, 128, 128), np.uint8),  # png
                'target_depth': ((None, 6, 128, 128), np.float32),
                'target_voxels': ((None, 1, 48, 48, 48), np.uint8),
            }
        else:
            raise NotImplementedError()

        self.total_num_examples = len(self.data_being_used)

        self.current_batch_index = 0
        self.batched_examples = None
        self.reset()

        # Defines all available field names, sizes, and datatypes.
        # TODO: use the values in self.all_data

        for fieldname in self.tensors_to_read:
            assert fieldname in self.available_fields, 'invalid field name {}'.format(fieldname)

    def reset(self):
        if self.shuffle:
            random.shuffle(self.data_being_used)
        self.batched_examples = list(make_chunks(self.data_being_used, n=self.batch_size))
        self.current_batch_index = 0

    def next(self):
        """
        last batch size may not be the same.
        :return: (examples, tensors) or None
        """
        if self.current_batch_index >= len(self.batched_examples):
            return None

        batch = self.batched_examples[self.current_batch_index]
        self.current_batch_index += 1

        tensors = {}
        for fieldname in self.tensors_to_read:
            shape, dtype = self.available_fields[fieldname]
            filenames = get_field_filenames2(batch, fieldname)
            assert shape[0] is None
            shape = (len(batch),) + shape[1:]
            tensors[fieldname] = parallel_reader.read_batch(filenames, shape=shape, dtype=dtype)

        return batch, tensors


def main():
    """
    Example usage.
    """
    loader = ExampleLoader('/data/mvshape/out/splits/shrec12_examples_vpo/train.bin',
                           ['single_depth', 'multiview_depth'],
                           batch_size=100,
                           shuffle=True)

    category_ids = set()
    while True:
        print(loader.current_batch_index)
        out = loader.next()
        if out is None:
            loader.reset()
            continue
        examples, tensors = out
        print(len(examples))
        print('category_id', examples[0].single_depth.category_id)
        print('category_count', len(category_ids))
        print('categories', sorted(list(category_ids)))
        category_ids.add(examples[0].single_depth.category_id)


def shrec12_convert_category_ids(ids):
    new_ids = shrec12_train_category_ids.searchsorted(ids)
    return new_ids


def prepare_data(next_batch):
    """
    :param next_batch: Output from data loader.
    :return: A dict of (field -> numpy array)
    """

    # A list of Example protobufs and a dict of tensors.
    examples, tensors = next_batch

    # prepare data
    # modify in-place.
    in_depth = tensors['single_depth'].copy()
    in_silhouette = np.isfinite(in_depth)
    in_depth[~in_silhouette] = 0
    in_silhouette = in_silhouette.astype(np.float32)
    in_image = np.concatenate([in_depth, in_silhouette], axis=1)
    assert (in_image.ndim == 4 and in_image.shape[1] == 2), in_image.shape

    target_depth = tensors['multiview_depth'].copy()
    target_silhouette = np.isfinite(target_depth)
    target_depth[~target_silhouette] = 0
    target_silhouette = target_silhouette.astype(np.float32)  # 0 or 1
    # both (b, 6, 128, 128)

    target_depth = np.expand_dims(target_depth, axis=2)
    target_silhouette = np.expand_dims(target_silhouette, axis=2)
    # both (b, 6, 1, 128, 128)

    ## config for the front-back depth model. flip the back depth images.
    target_depth[:, 1::2] = target_depth[:, 1::2, :, :, ::-1]
    # (b, 6, 1, 128, 128)
    target_silhouette = target_silhouette[:, ::2]
    # (b, 3, 1, 128, 128)

    categories = np.array(shrec12_convert_category_ids([example.single_depth.category_id - 1 for example in examples]), dtype=np.int64)

    return {
        'in_image': in_image,
        'in_depth': in_depth,
        'target_depth': target_depth,
        'target_silhouette': target_silhouette,
        'categories': categories,
    }


def to_masked_image(depth, silhouette, fill_value=np.nan):
    assert isinstance(depth, np.ndarray)
    assert isinstance(silhouette, np.ndarray)
    assert depth.shape == silhouette.shape
    assert silhouette.ndim >= 2

    out = depth.copy()

    if silhouette.dtype == np.float32:
        binary_silhouette = silhouette > 0
    elif silhouette.dtype == np.bool:
        binary_silhouette = silhouette
    else:
        raise NotImplemented(type(silhouette))

    out[~binary_silhouette] = fill_value

    return out


if __name__ == '__main__':
    main()
