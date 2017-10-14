import time
import bz2
import pickle
from dshin import io_utils


def load_saved_output(filename):
    with bz2.BZ2File(filename, 'rb') as f:
        ret = pickle.load(f)
    assert isinstance(ret, dict)
    return ret


class Shrec12Qualitative(object):
    def __init__(self):
        pass

    def mesh(self, name, experiment, index):
        filename = '/data/daeyun/shrec12_recon_links/{}/{:03d}/{}.off'.format(experiment, index, name)
        return io_utils.read_off(filename)

    def values(self, experiment, index):
        filename1 = '/data/daeyun/shrec12_recon_links/{}/{:03d}/out_images.pkl.bz2'.format(experiment, index)
        filename2 = '/data/daeyun/shrec12_recon_links/{}/{:03d}/out_voxels.pkl.bz2'.format(experiment, index)
        ret1 = load_saved_output(filename1)
        ret2 = load_saved_output(filename2)
        ret1.update(ret2)
        return ret1
