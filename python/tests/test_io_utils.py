import os
import numpy as np
from os import path
from mvshape import io_utils


def test_read_png():
    filename = io_utils.make_resources_path('render_for_cnn/syn_images_cropped/02691156/1c87b854cceb778615aa5b227c027ee0/02691156_1c87b854cceb778615aa5b227c027ee0_v0003_a044_e001_t003_d005.png')
    assert path.exists(filename)
    image = io_utils.read_png(filename)
    assert image.ndim == 3
    assert image.shape[2] == 4


def test_save_read_identity():
    for _ in range(10):
        randimg = np.random.randn(np.random.randint(1, 14), np.random.randint(1, 14), 4)
        randimg /= randimg.max()
        randimg[randimg < 0] = 0
        if np.random.randint(0, 2) == 0:
            randimg *= 255
        else:
            randimg *= 100
            randimg += 50
        randimg = randimg.astype(np.uint8)

        temp_file = io_utils.temp_filename(suffix='.png')

        io_utils.save_png(randimg, filename=temp_file)

        loaded_file = io_utils.read_png(temp_file)

        assert loaded_file.dtype == np.uint8
        assert loaded_file.shape == randimg.shape
        assert (loaded_file == randimg).all()

        os.remove(temp_file)
