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
import typing
import blosc
import numpy.linalg as la

from dshin import log
from dshin import transforms


def centroid_and_distance_from_centroid(pts):
    centroid = pts.mean(0)
    distance = la.norm(pts - centroid, axis=1).mean()
    return centroid, distance


def find_aligning_transformation(source, target) -> typing.Tuple[np.ndarray, float]:
    target_mean, target_dist = centroid_and_distance_from_centroid(target)
    source_mean, source_dist = centroid_and_distance_from_centroid(source)

    # transformed_source = (source - source_mean) * target_dist / source_dist + target_mean
    # transformed_source = source * target_dist / source_dist - source_mean * target_dist / source_dist + target_mean
    xyz_offset = target_mean - source_mean * target_dist / source_dist
    scale = target_dist / source_dist

    return xyz_offset, scale
