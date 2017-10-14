import contextlib
import copy
import errno
import fcntl
import hashlib
import json
import numbers
import os
import sys
import tempfile
import time
import struct
from os import path

import numpy as np
import plyfile
import stl
import ujson
import blosc
import numpy.linalg as la

from dshin import log
from dshin import transforms


def transform_ply(filename, out_filename, mat34: np.ndarray, confidence_scale=None, value_scale=None):
    plydata = plyfile.PlyData.read(filename)
    assert 'vertex' in plydata and plydata['vertex']['x'].size > 0
    v = np.vstack((plydata['vertex']['x'], plydata['vertex']['y'],
                   plydata['vertex']['z'])).T.astype(np.float64)
    n = np.vstack((plydata['vertex']['nx'], plydata['vertex']['ny'],
                   plydata['vertex']['nz'])).T.astype(np.float64)
    vn = v + n

    v_ = transforms.apply34(mat34, v)
    vn_ = transforms.apply34(mat34, vn)
    n_ = vn_ - v_

    plydata['vertex']['x'] = v_[:, 0]
    plydata['vertex']['y'] = v_[:, 1]
    plydata['vertex']['z'] = v_[:, 2]

    plydata['vertex']['nx'] = n_[:, 0]
    plydata['vertex']['ny'] = n_[:, 1]
    plydata['vertex']['nz'] = n_[:, 2]

    if confidence_scale is not None:
        assert isinstance(confidence_scale, float)
        plydata['vertex']['confidence'] *= confidence_scale
        plydata['vertex']['confidence'] = np.minimum(plydata['vertex']['confidence'], 1.0)

    if value_scale is not None:
        assert isinstance(value_scale, float)
        plydata['vertex']['value'] *= value_scale



    plydata.write(out_filename)

    # Replace \r\n with \n.
    with open(out_filename, 'rb') as f:
        content = f.read()
    beginning = content[:128]
    rest = content[128:]
    beginning = beginning.replace(b'ply\r\n', b'ply\n')

    with open(out_filename, 'wb') as f:
        f.write(beginning + rest)

    return plydata
