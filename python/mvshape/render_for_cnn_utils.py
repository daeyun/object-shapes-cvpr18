import math
import random
import re
import time
import os
from os import path
from mvshape import io_utils
import glob
import numpy as np
from dshin import log
from dshin import transforms
import multiprocessing as mp
import scipy.misc


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
    relative_std = 0.03

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

    # synset filtering
    new_filenames = []
    for filename in filenames:
        m = re.search(r'syn_images_cropped/(.*?)/(.*?)/[^_]+?_[^_]+?_v(\d{4,})_a', filename)
        synset = m.group(1)
        if synset not in ['02858304', '02876657', '02924116', '04379243']:
            continue
        new_filenames.append(filename)
    filenames = new_filenames
    assert len(filenames) > 100
    print('Number of images to resize:', len(filenames))

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


def get_Rt_from_RenderForCNN_parameters(params):
    """
    Computes a camera matrix that transforms models coordinates to blender's camera coordinates given RenderForCNN camera parameters.

    Transformation steps in blender:
        ((model coordinates, blender import config) -> blender world coordinates, params) -> blender camera coordinates

    This function:
        (model coordinates, params) -> blender camera coordinates.

    :param params: azimuth, elevation, tilt, distance from origin. All angles are in degrees.
    :return: (3, 4) camera matrix.
    """
    assert len(params) == 4
    assert not isinstance(params[0], (np.ndarray, list, tuple))

    # Same ordering used in RenderForCNN.
    azimuth_deg, elevation_deg, theta_deg, rho = params

    cam_xyz = transforms.sph_to_xyz([rho, 90 - elevation_deg, azimuth_deg], is_input_radians=False)

    # Blender's default obj import config:
    # up (z):  +y forward (y): -z
    cam_xyz[1], cam_xyz[2] = cam_xyz[2], -cam_xyz[1]

    # In model coordinates, default up vector is y+
    # "lookat" position is assumed to be (0,0,0)
    M0 = transforms.lookat_matrix(cam_xyz=cam_xyz, obj_xyz=(0, 0, 0), up=(0, 1, 0))

    M = np.eye(4)
    M[:3, :] = M0
    M0 = M

    T_tilt = transforms.rotation_matrix(-theta_deg, direction=np.array((0, 0, 1)), deg=True)

    # Rt: move camera then tilt in camera space.
    M = T_tilt.dot(M0)
    Rt = M[:3]

    assert Rt.shape == (3, 4)
    assert Rt.dtype == np.float64

    return Rt


def read_params_file(params_txt_filename):
    with open(params_txt_filename, 'r') as f:
        content = f.read()
    lines = [[float(item) for item in re.split(r'\s+', line.strip())] for line in re.split(r'\r?\n', content) if line]
    return lines


def _load_image(filename):
    img = io_utils.read_jpg(filename)
    return img


def blend_and_resize(params):
    object_image_filename = params['object_image_filename']
    bkg_filenames = params['bkg_filenames']
    out_image_filename = params['out_image_filename']

    object_image = io_utils.read_png(object_image_filename)

    resolution = 128
    bkg_clutter_ratio = 0.8
    scale_max = 4

    use_background_image = random.random() < bkg_clutter_ratio
    resize_method = {0: 'bilinear',
                     1: 'bicubic',
                     2: 'lanczos'}[random.randint(0, 2)]

    def force_uint8(arr):
        if arr.dtype in (np.float32, np.float64):
            arr = (arr * 255).round().astype(np.uint8)
        assert arr.dtype == np.uint8
        return arr

    def force_float(arr):
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        elif arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        assert arr.dtype == np.float32
        return arr

    def resize(arr: np.ndarray, res):
        assert arr.shape[0] == arr.shape[1]
        arr = force_uint8(arr)
        resized = scipy.misc.imresize(arr, size=(res, res), interp=resize_method)
        return force_float(resized)

    # Crop and pad.
    iy, ix = np.where(object_image)[:2]
    y0 = np.min(iy)
    x0 = np.min(ix)
    y1 = np.max(iy) + 1
    x1 = np.max(ix) + 1
    src_cropped = object_image[y0:y1, x0:x1]
    src_cropped_square = make_randomized_square_image(src_cropped)

    src_s = src_cropped_square.shape[0]
    target_resolution = min(src_s, resolution)

    if use_background_image:
        # Read random file.
        bkg_image = None
        while True:
            bkg_image = _load_image(random.choice(bkg_filenames))
            s = min(bkg_image.shape[:2])
            if s >= target_resolution:
                break

        if bkg_image.ndim == 2:
            bkg_image = np.tile(bkg_image[:, :, None], (1, 1, 3))

        # resize background image. if the iamge is smaller than 128, don't resize.
        res = random.randint(target_resolution, min(target_resolution * scale_max, min(bkg_image.shape[:2])))
        y = random.randint(0, bkg_image.shape[0] - res)
        x = random.randint(0, bkg_image.shape[1] - res)
        bkg_cropped = bkg_image[y:y + res, x:x + res]
        assert bkg_cropped.shape[0] == bkg_cropped.shape[1]
        assert bkg_cropped.shape[0] == res
        assert bkg_cropped.shape[2] == 3

        bkg = resize(bkg_cropped, res=target_resolution)
    else:
        color_gray = random.random()
        bkg = color_gray * np.ones((target_resolution, target_resolution, 3), dtype=np.float32)

    src_image_rgba = resize(src_cropped_square, res=target_resolution)
    src_image = src_image_rgba[:, :, :3]
    mask = src_image_rgba[:, :, 3]

    assert bkg.dtype == np.float32
    assert bkg.shape[0] == bkg.shape[1]
    assert bkg.shape[2] == 3
    assert bkg.shape[0] == target_resolution
    assert src_image.dtype == np.float32
    assert src_image.shape == bkg.shape
    assert mask.dtype == np.float32

    blended = ((1.0 - mask)[:, :, None] * bkg) + ((mask)[:, :, None] * src_image)

    blended_final = force_uint8(resize(blended, res=resolution))

    assert blended_final.shape[0] == blended_final.shape[1]
    assert blended_final.shape[0] == resolution
    assert blended_final.dtype == np.uint8

    io_utils.ensure_dir_exists(path.dirname(out_image_filename), log_mkdir=False)
    scipy.misc.imsave(out_image_filename, blended_final)

    return out_image_filename
