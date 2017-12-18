import mvshape.data
import mvshape.data.parallel_reader
import numpy as np
import random
import glob
import os
from os import path
import time

base = path.expanduser('/data/mvshape/')


def test_png():
    count = 0
    iter = os.scandir(path.join(base, 'shapenetcore/single_rgb_128/'))
    elapsed_times = []
    while True:
        files = []
        while True:
            entry = next(iter)
            files.append(entry.path.encode('utf-8'))
            if len(files) >= 200:
                break
        start = time.time()
        b = mvshape.data.parallel_reader.read_batch(files, shape=(200, 3, 128, 128), dtype=np.uint8)
        elapsed = time.time() - start
        elapsed_times.append(elapsed)
        print(count, b.shape)
        count += 1
        if count % 100 == 0:
            print('#############', np.median(elapsed_times))
            elapsed_times = []


if __name__ == '__main__':
    test_png()
