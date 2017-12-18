import ctypes
import logging
from sys import platform
import shutil
from ctypes import cdll
from os import path
from dshin import log
from mvshape import io_utils
import numpy as np

from numpy.ctypeslib import ndpointer
import mvshape

proj_root = mvshape.proj_root
lib_filename = mvshape.make_bin_path('cpp/ctypes/librender_object.so')
assert path.isfile(lib_filename), lib_filename

l = cdll.LoadLibrary(lib_filename)
l.depth_and_normal_map.restype = ctypes.c_int
l.depth_and_normal_map.argtypes = [
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ctypes.c_size_t,
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_char_p,
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
]
lib = l

log.info('Loaded shared library {}'.format(lib._name))


def depth_and_normal_map(vertices, eye, center, up, wh):
    tri = vertices.astype(np.float32)
    eye = np.array(eye, np.float32)
    center = np.array(center, np.float32)
    up = np.array(up, np.float32)

    rendered_depth = np.zeros([wh[0], wh[1]], dtype=np.float32)
    rendered_normal = np.zeros([wh[0], wh[1], 3], dtype=np.float32)

    resources_dir = path.realpath(path.join(proj_root, 'resources'))
    lib.depth_and_normal_map(tri, tri.size, eye, center, up, 128, 128,
                             ctypes.c_char_p(resources_dir.encode('ascii')),
                             rendered_depth,
                             rendered_normal)

    return rendered_depth


def main():
    fv = io_utils.read_mesh('/data/mvshape/mesh/shrec12/Train1/Bookset_aligned/D01013_out.off')
    tri = fv['v'][fv['f']]  # type: np.ndarray

    pass


if __name__ == '__main__':
    main()
