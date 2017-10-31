import shutil
import numpy.linalg as la
import math
import random
import hashlib
from os import path
from pprint import pprint
import re
import glob
import numpy as np
import time
from dshin import transforms
from dshin import camera
from mvshape import io_utils
from mvshape import db_model as dbm
from mvshape import render_for_cnn_utils
from os.path import join
from dshin import log
import scipy.misc

is_subset = True


def main():
    log.info('Getting all filenames.')
    syndirs = sorted(glob.glob('/data/render_for_cnn/data/syn_images_cropped_bkg_overlaid/*'))
    random.seed(42)
    filenames = []
    for syndir in syndirs:
        modeldirs = sorted(glob.glob(path.join(syndir, '*')))
        if is_subset:
            modeldirs = modeldirs[:10]
        for modeldir in modeldirs:
            renderings = sorted(glob.glob(path.join(modeldir, '*')))
            if is_subset:
                renderings = renderings[:7]
            filenames.extend(renderings)

    log.info('{} files'.format(len(filenames)))

    data_base_dir = '/data/mvshape'

    start_time = time.time()

    random.seed(42)
    log.info('Processing rgb images.')
    for i, filename in enumerate(filenames):
        m = re.search(r'syn_images_cropped_bkg_overlaid/(.*?)/(.*?)/[^_]+?_[^_]+?_v(\d{4,})_a', filename)
        synset = m.group(1)
        model_name = m.group(2)
        v = m.group(3)
        image_num = int(v)

        vc_rendering_name = '{}_{:04d}'.format(model_name, image_num)
        out_filename = path.join(data_base_dir, 'out/shapenetcore/single_rgb_128/{}.png'.format(vc_rendering_name))

        assert path.isfile(filename)
        io_utils.ensure_dir_exists(path.dirname(out_filename))

        img = io_utils.read_jpg(filename)
        assert img.shape[0] == img.shape[1]
        assert img.shape[2] == 3

        resize_method = {0: 'bilinear',
                         1: 'bicubic',
                         2: 'lanczos'}[random.randint(0, 2)]

        resized_img = scipy.misc.imresize(img, (128, 128), interp=resize_method)
        assert resized_img.dtype == np.uint8

        # io_utils.save_array_compressed(out_filename, resized_img)
        scipy.misc.imsave(out_filename, resized_img)

        if i % 100 == 0:
            t_elapsed = (time.time() - start_time)
            t_remaining = (t_elapsed / (i + 1) * (len(filenames) - i))
            log.info('Creating examples in db. {} of {}. elapsed: {:.1f} min, remaining: {:.1f} min'.format(i, len(filenames), t_elapsed / 60, t_remaining / 60))

    t_elapsed = (time.time() - start_time)
    log.info('total elapsed: {:.1f} min'.format(t_elapsed / 60))




if __name__ == '__main__':
    main()
