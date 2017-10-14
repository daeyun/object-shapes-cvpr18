import pickle

import time
from dshin import geom2d
from dshin import plot_utils
from dshin import io_utils

from multiview_shape.data import cvpr15_shrec12
from multiview_shape import db_model as dbm
from dshin import plot_utils

from pprint import pprint


def main():
    dbm.init('/data/multiview_shape.sqlite')
    ds = cvpr15_shrec12.DataSource('/data/multiview_shape_sql_data/', is_seamless=False, load_tf_records=True)

    filenames = {}
    depth_20 = {}
    depth = {}
    voxels = {}

    key = ['test', 'novelview']
    b = ds.next(key, batch_size=600)
    filenames['novelview'] = b['mesh']
    depth_20['novelview'] = b['depth_20']
    depth['novelview'] = b['depth']
    voxels['novelview'] = b['voxels']

    key = ['test', 'novelmodel']
    b = ds.next(key, batch_size=600)
    filenames['novelmodel'] = b['mesh']
    depth_20['novelmodel'] = b['depth_20']
    depth['novelmodel'] = b['depth']
    voxels['novelmodel'] = b['voxels']

    key = ['test', 'novelclass']
    b = ds.next(key, batch_size=600)
    filenames['novelclass'] = b['mesh']
    depth_20['novelclass'] = b['depth_20']
    depth['novelclass'] = b['depth']
    voxels['novelclass'] = b['voxels']

    data = {}
    data['filenames'] = filenames
    data['depth_20'] = depth_20
    data['depth'] = depth
    data['voxels'] = voxels

    with open('/data/daeyun/shrec12_recon_links/gt.pkl', 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    main()
