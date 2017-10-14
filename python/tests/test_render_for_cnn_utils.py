import os
import glob
import numpy as np
from os import path
from mvshape import io_utils
from mvshape import render_for_cnn_utils


def test_make_square_image_png():
    filename = io_utils.make_resources_path('render_for_cnn/syn_images_cropped/02691156/1c87b854cceb778615aa5b227c027ee0/02691156_1c87b854cceb778615aa5b227c027ee0_v0003_a044_e001_t003_d005.png')
    image = io_utils.read_png(filename)
    new_image = render_for_cnn_utils.make_square_image(image)

    assert new_image.shape[0] == new_image.shape[1]
    assert new_image.shape[2] == 4
    assert new_image.dtype == image.dtype


def test_make_square_image_random_sizes():
    for _ in range(500):
        randimg = np.random.randn(np.random.randint(1, 10), np.random.randint(1, 10), 4)
        randimg /= randimg.max()
        randimg[randimg < 0] = 0
        randimg *= 255
        randimg = randimg.astype(np.uint8)

        new_image = render_for_cnn_utils.make_square_image(randimg, fill_value=0)
        assert new_image.shape[0] == new_image.shape[1]
        assert new_image.shape[0] == max(randimg.shape[:2])
        assert new_image.shape[2] == 4
        assert new_image.dtype == randimg.dtype


def test_truncated_square_images():
    filenames = glob.glob(io_utils.make_resources_path('render_for_cnn/syn_images_cropped/*/*/*.png'))
    assert len(filenames) > 5
    temp_file = io_utils.temp_filename(suffix='.png')
    for filename in filenames:
        image = io_utils.read_png(filename)
        truncated = render_for_cnn_utils.make_randomized_square_image(image)
        assert truncated.shape[0] == truncated.shape[0]
        assert truncated.shape[2] == 4
        assert truncated.dtype == np.uint8
        io_utils.save_png(truncated, temp_file)
        loaded = io_utils.read_png(temp_file)
        assert (truncated == loaded).all()
        assert truncated.dtype == loaded.dtype
    os.remove(temp_file)
