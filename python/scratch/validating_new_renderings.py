import glob
import shutil
import random
from os import path

import sys
import hashlib

BUF_SIZE = 65536


def hash_file(filename):
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()


def main():
    original = "/data/mvshape/shrec12/"
    new = "/data/mvshape/out/shrec12/"

    original_files = glob.glob(path.join(original, '**/*.bin'))
    new_files = glob.glob(path.join(new, '**/*.bin'))

    a = set([item[len(original):] for item in original_files])
    b = set([item[len(new):] for item in new_files])

    print('original size ', len(a))
    print('new size ', len(b))
    print('union size ', len(a.union(b)))
    print('difference size ', len(b.difference(a)))

    a = list(a)
    random.shuffle(a)
    a_subset = a[:1000]

    for item in a_subset:
        a_hash = hash_file(path.join(original, item))
        b_hash = hash_file(path.join(new, item))
        assert a_hash == b_hash

    diff = list(b.difference(a))
    for item in diff:
        assert path.exists(path.join(new, item))
        assert not path.exists(path.join(original, item))

        shutil.copy(path.join(new, item), path.join(original, item))
        print(path.join(original, item))






if __name__ == '__main__':
    main()
