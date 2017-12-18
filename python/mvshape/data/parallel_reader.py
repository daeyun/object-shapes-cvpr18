import ctypes
from os import path
from sys import platform
from ctypes import cdll

import numpy as np
from numpy.ctypeslib import ndpointer

from dshin import log

ctypes_lib_dirname = path.realpath(path.join(path.dirname(__file__), '../../../cmake-build-release/cpp/ctypes/'))

lib = None
if platform == "linux" or platform == "linux2":
    lib_filename = path.join(ctypes_lib_dirname, 'libsingle_batch_loader_ctypes.so')
    assert path.isfile(lib_filename), 'file does not exist: {}'.format(lib_filename)
    lib = cdll.LoadLibrary(lib_filename)
else:
    raise NotImplemented(platform)

variations = {
    np.float32: ("ReadSingleBatch_float32", ctypes.c_float),
    np.int32: ("ReadSingleBatch_int32", ctypes.c_int32),
    np.uint8: ("ReadSingleBatch_uint8", ctypes.c_uint8),
}

if lib:
    for c_func_name, ctypes_dtype in variations.values():
        c_func = getattr(lib, c_func_name)
        c_func.restype = ctypes.c_uint32
        c_func.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ndpointer(ctypes_dtype, flags="C_CONTIGUOUS"),
        ]
    log.info('Loaded shared library %s', lib._name)


def read_batch(filenames, shape, dtype=np.float32):
    """
    Reads tensors in parallel. The order of filenames is preserved.
    Releases GIL.

    :param filenames: A list of N paths to compressed tensor data files to read.
    :param shape: (N, ...)
    :param dtype: A NumPy data type.
    :return: The concatenated array/tensor.
    """
    assert len(filenames) == shape[0], (len(filenames), shape)
    assert isinstance(filenames[0], bytes)

    if dtype not in variations:
        raise RuntimeError("Invalid dtype: " + str(dtype))

    c_func_name, _ = variations[dtype]

    out_data = np.empty(shape, dtype=dtype)
    assert out_data.flags['C_CONTIGUOUS']

    filenames_p = (ctypes.c_char_p * len(filenames))()
    filenames_p[:] = filenames
    shape_p = (ctypes.c_int32 * len(out_data.shape))()
    shape_p[:] = out_data.shape

    c_func = getattr(lib, c_func_name)

    # Releases GIL, creates and joins threads.
    c_func(filenames_p, len(filenames), shape_p, len(out_data.shape), out_data)

    return out_data
