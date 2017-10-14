import os
from os import path

data_root = '/data/mvshape'
bin_root = path.realpath(path.join(path.dirname(__file__), '../../cmake-build-release'))
proj_root = path.realpath(path.join(path.dirname(__file__), '../..'))


def _ensure_no_leading_path_separator(p):
    if p[0] == '/':
        p = p[1:]
    return p


def make_data_path(p):
    return path.join(data_root, _ensure_no_leading_path_separator(p))


def make_bin_path(p):
    return path.join(bin_root, _ensure_no_leading_path_separator(p))
