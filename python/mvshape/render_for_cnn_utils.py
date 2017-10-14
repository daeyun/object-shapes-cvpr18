import math
import time
import os
from os import path
from mvshape import io_utils
import glob
import numpy as np
from dshin import log
import multiprocessing as mp


def make_square_image(image: np.ndarray, fill_value=0):
    """
    Pads the image with `fill_value` so that the padded image is square-shaped.
    :param image:
    :param fill_value:
    :return:
    """
    assert image.ndim == 3
    assert image.shape[2] == 4, 'Only RGBA images are supported for now.'

    argmax = np.argmax(image.shape[:2]).item()

    smax = image.shape[argmax]
    smin = image.shape[1 - argmax]

    padding = (smax - smin) / 2
    before = math.floor(padding)
    after = math.ceil(padding)

    if argmax == 0:
        pad_width = [(0, 0), (before, after), (0, 0)]
    elif argmax == 1:
        pad_width = [(before, after), (0, 0), (0, 0)]
    else:
        raise RuntimeError()

    new_image = np.pad(image, pad_width, mode='constant', constant_values=fill_value)
    assert new_image.shape[0] == new_image.shape[1]
    assert new_image.shape[2] == 4

    return new_image


def pad_or_crop_image(image: np.ndarray, before_and_afters, fill_value=0):
    assert image.ndim == 3
    assert image.shape[2] == 4, 'Only RGBA images are supported for now.'
    assert image.ndim == len(before_and_afters)

    for i, (before, after) in enumerate(before_and_afters):
        if image.shape[i] <= -(min(0, before) + min(0, after)):
            raise ValueError('Cropping too much.')

    crop_params = [
        slice(max(0, -before), min(image.shape[i], image.shape[i] + after)) for i, (before, after) in enumerate(before_and_afters)
    ]

    cropped = image[crop_params]

    pad_params = [
        (max(0, before), max(0, after)) for before, after in before_and_afters
    ]

    padded = np.pad(cropped, pad_params, mode='constant', constant_values=fill_value)

    return padded


def random_pad_or_crop_image(image, relative_std=0.06):
    """

    :param image:
    :param relative_std: 0.1 means 1 standard deviation corresponds to a 10% width difference.
    :return:
    """

    params = []
    for i in range(image.ndim):
        before, after = (np.random.randn(2) * relative_std * image.shape[i]).astype(np.int32)
        params.append((before, after))

    ret = pad_or_crop_image(image, params, fill_value=0)
    return ret


def make_randomized_square_image(image):
    relative_std = 0.07

    # <hack>
    # Making sure no cropping happens in the dimension that will be padded to make the image square.
    redo_count = 0
    while True:
        new_image_shape = []
        params = []
        assert image.ndim == 3
        for i in range(image.ndim):
            if i == image.ndim - 1:
                before, after = 0, 0  # No padding in the channel dimension.
            else:
                before, after = (np.random.randn(2) * relative_std * image.shape[i]).astype(np.int32)

            params.append((before, after))
            new_image_shape.append(image.shape[i] + before + after)
        argmax = np.argmax(new_image_shape[:2])
        redo = False
        if argmax == 0:
            if (np.array(params[1]) < 0).any():
                redo = True
        elif argmax == 1:
            if (np.array(params[0]) < 0).any():
                redo = True
        else:
            raise RuntimeError()
        if not redo:
            break
        redo_count += 1
        if redo_count > 1000:
            raise RuntimeError('This implementation is probably wrong.')
    if redo_count > 50:
        log.warn('slow implementation. redo count: {}'.format(redo_count))
    # </hack>

    new_image = pad_or_crop_image(image, params, fill_value=0)

    assert new_image.shape == tuple(new_image_shape)

    new_image = make_square_image(new_image)

    return new_image


def _truncate_images_worker(params):
    filename, out_dir, ignore_overwrite = params['filename'], params['out_dir'], params['ignore_overwrite']
    filename_parts = filename.split(os.sep)
    path_suffix = os.sep.join(filename_parts[-3:])
    out_filename = os.path.join(out_dir, path_suffix)
    if ignore_overwrite and os.path.isfile(out_filename):
        # Skip if file already exists.
        return None

    image = io_utils.read_png(filename)
    truncated = make_randomized_square_image(image)
    assert truncated.shape[0] == truncated.shape[0]
    assert truncated.shape[2] == 4
    assert truncated.dtype == np.uint8

    out_filename_parent_dir = os.path.dirname(out_filename)
    io_utils.ensure_dir_exists(out_filename_parent_dir, log_mkdir=False)

    io_utils.save_png(truncated, out_filename)

    return out_filename


def truncate_images(syn_images_cropped_exact_dir, out_dir, ignore_overwrite=True):
    log.info('looking for png files in {}'.format(syn_images_cropped_exact_dir))
    assert os.path.isdir(syn_images_cropped_exact_dir)
    filenames = sorted(list(glob.glob(path.join(syn_images_cropped_exact_dir, '*/*/*.png'))))
    assert len(filenames) > 0
    log.info('found {} images'.format(len(filenames)))

    start_time = time.time()

    params = [
        {
            'filename': filename,
            'out_dir': out_dir,
            'ignore_overwrite': ignore_overwrite,
        } for filename in filenames
    ]

    pool = mp.Pool()

    for i, out_filename in enumerate(pool.imap(_truncate_images_worker, params, chunksize=200)):
        if i % 2000 == 0:
            elapsed = time.time() - start_time
            remaining = elapsed / (i + 1) * (len(filenames) - (i + 1))
            log.info('{} of {}. Elapsed: {} min. Remaining: {} min'.format(i, len(filenames), int(elapsed / 60), int(remaining / 60)))
