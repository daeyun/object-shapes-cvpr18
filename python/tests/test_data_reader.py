import random
from os import path

import numpy as np

from mvshape import eval_out_reader
from mvshape.data import parallel_reader


def test_parallel_read():
    split = path.realpath(path.join(path.dirname(__file__), '../../resources/data/splits/shrec12_examples_vpo/test.bin'))
    metadata = eval_out_reader.read_split_metadata(split)
    base = path.realpath(path.join(path.dirname(__file__), '../../resources/data'))

    random.shuffle(metadata)

    filenames = []
    for item in metadata:
        name = item.multiview_depth.filename
        filename = path.join(base, name[1:])
        filenames.append(filename.encode('utf-8'))

    shape = (len(filenames), item.multiview_depth.set_size, 128, 128, 1)

    data = parallel_reader.read_batch(filenames, shape, dtype=np.float32)

    assert np.isnan(data.ravel()[0])
    assert np.sum(~np.isnan(data[0])) / np.sum(np.isnan(data[0])) > 0.02
