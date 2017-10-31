from os import path
import os
import time
from pprint import pprint
from dshin import log
import multiprocessing as mp
import glob
import random
from mvshape import render_for_cnn_utils


def main():
    # syn_images_dir = '/data/render_for_cnn/data/syn_images/'
    # out_dir = '/tmp/hi'

    syn_images_dir = '/data/syn_images/'
    out_dir = '/data/syn_images_final/'

    assert path.isdir(syn_images_dir)

    bkg_dir = '/home/daeyun/git/RenderForCNN/datasets/sun2012pascalformat/JPEGImages'
    bkg_file_list = '/home/daeyun/git/RenderForCNN/datasets/sun2012pascalformat/filelist.txt'

    syn_images_filenames = glob.glob(path.join(syn_images_dir, '**/*.png'), recursive=True)
    syn_images_filenames = sorted(syn_images_filenames)

    with open(bkg_file_list, 'r') as f:
        content = f.read()
    fnames = content.strip().split()
    background_filenames = [path.join(bkg_dir, name) for name in fnames]

    assert path.isfile(syn_images_filenames[0])
    assert path.isfile(background_filenames[0])

    log.info('{} object images found'.format(len(syn_images_filenames)))
    log.info('{} background images found'.format(len(background_filenames)))

    start_time = time.time()

    all_params = [
        {
            'object_image_filename': filename,
            'bkg_filenames': background_filenames,
            'out_image_filename': path.join(out_dir, '/'.join(filename.split('/')[-3:]))
        } for filename in syn_images_filenames
    ]

    pool = mp.Pool(processes=os.cpu_count() * 2)

    for i, out_filename in enumerate(pool.imap(render_for_cnn_utils.blend_and_resize, all_params, chunksize=200)):
        if i % 2000 == 0:
            elapsed = time.time() - start_time
            remaining = elapsed / (i + 1) * (len(all_params) - (i + 1))
            log.info('{} of {}. Elapsed: {} min. Remaining: {} min'.format(i, len(all_params), int(elapsed / 60), int(remaining / 60)))


if __name__ == '__main__':
    main()
