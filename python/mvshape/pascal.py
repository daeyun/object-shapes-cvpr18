import scipy.io as sio
from dshin import transforms
import numpy as np
import scipy.misc as sp_misc
from os import path
from mvshape import io_utils
import math
from dshin import camera
import numpy.linalg as la


# same as the one from render_for_cnn_utils.py
def make_square_image(image: np.ndarray, fill_value=None):
    """
    Pads the image with `fill_value` so that the padded image is square-shaped.
    :param image:
    :param fill_value:
    :return:
    """
    assert image.ndim == 3
    # assert image.shape[2] == 4, 'Only RGBA images are supported for now.'

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

    if fill_value is None:
        new_image = np.pad(image, pad_width, mode='edge')
    else:
        new_image = np.pad(image, pad_width, mode='constant', constant_values=fill_value)
    assert new_image.shape[0] == new_image.shape[1]
    # assert new_image.shape[2] == 4

    return new_image


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


def pascal_to_shapenet_mesh_transformation():
    # swap axes
    R1 = transforms.xrotate(-90)
    R2 = transforms.yrotate(90)
    R = R2.dot(R1)
    return R


class PascalAnnotation(object):
    def __init__(self, matfile: str):
        assert matfile.endswith('.mat')
        assert path.isfile(matfile)
        m = sio.loadmat(matfile)
        self.matfile = matfile
        assert (len(m['record']) == 1)
        assert (len(m['record']) == 1)
        assert (len(m['record'][0]['filename']) == 1)

        record = m['record'][0]
        s = record['filename'][0][0]
        assert (len(record['filename']) == 1)
        assert (len(record['filename'][0]) == 1)

        self.jpg_filename = s.encode().decode('utf-8')
        dirname = self.matfile.split('/')[-2]
        dataset_base = '/'.join(self.matfile.split('/')[:-3])
        self.jpg_fullpath = path.join(dataset_base, 'Images', dirname, self.jpg_filename)
        assert path.isfile(self.jpg_fullpath)

        assert len(record['objects'][0]) == 1

        num_objects = len(record['objects'][0][0])
        self.objects = [record['objects'][0][0][i] for i in range(num_objects)]
        self.num_objects = num_objects

        # Using only the first object for now
        self.bbox = record['objects'][0][0]['bbox'][0][0].astype(np.int).tolist()
        self.image_hw = (record['size'][0][0]['height'][0][0].item(), record['size'][0][0]['width'][0][0].item())
        self.xstart, self.ystart, self.xend, self.yend = self.bbox

        self.category_name = record['objects'][0][0]['class'][0][0].encode().decode('utf-8')
        assert self.category_name in self.jpg_fullpath

        self.azimuth = self.objects[0]['viewpoint'][0][0]['azimuth'][0][0].item()
        self.elevation = self.objects[0]['viewpoint'][0][0]['elevation'][0][0].item()
        self.theta = self.objects[0]['viewpoint'][0][0]['theta'][0][0].item()
        self.distance = self.objects[0]['viewpoint'][0][0]['distance'][0][0].item()
        self.cad_index = record['objects'][0][0]['cad_index'][0][0].item()

        self.mesh_filename = path.join(dataset_base, 'CAD', self.category_name, '{:02d}.off'.format(self.cad_index))
        assert path.isfile(self.mesh_filename)

    def _imread(self):
        im = sp_misc.imread(self.jpg_fullpath)
        assert im.shape[:2] == self.image_hw
        return im

    def _cropped(self):
        im = self._imread()
        xstart, ystart, xend, yend = self.bbox

        ystart = max(ystart, 0)
        xstart = max(xstart, 0)

        w = xend - xstart
        h = yend - ystart
        if w > h:
            pad_top = (w - h) // 2
            pad_bottom = int(np.ceil((w - h) / 2.0).item())
            pad = min(min(ystart, pad_top), min(im.shape[0] - yend, pad_bottom))
            ystart -= pad
            yend += pad
        elif h > w:
            pad_left = (h - w) // 2
            pad_right = int(np.ceil((h - w) / 2.0).item())
            pad = min(min(xstart, pad_left), min(im.shape[1] - xend, pad_right))
            xstart -= pad
            xend += pad

        c = im[ystart:yend, xstart:xend]
        if c.ndim != 3:
            assert c.ndim == 2
            c = np.tile(c[:, :, None], [1, 1, 3])
        return c

    def square_image(self, res=128, fill_value=None):
        im = self._imread()
        xstart, ystart, xend, yend = self.bbox

        ystart = max(ystart, 0)
        xstart = max(xstart, 0)
        yend = min(yend, im.shape[0])
        xend = min(xend, im.shape[1])

        w = xend - xstart
        h = yend - ystart
        if w > h:
            pad_top = (w - h) // 2
            pad_bottom = int(np.ceil((w - h) / 2.0).item())
            # print(pad_top, pad_bottom)
            pad_top = min(ystart, pad_top)
            pad_bottom = min(im.shape[0] - yend, pad_bottom)
            ystart -= pad_top
            yend += pad_bottom
            pad_offset = pad_top - pad_bottom
            # print(pad_offset)
            if (w - h) % 2 == 1:
                pad_offset += 1
        elif h > w:
            pad_left = (h - w) // 2
            pad_right = int(np.ceil((h - w) / 2.0).item())
            # print(pad_left, pad_right)
            pad_left = min(xstart, pad_left)
            pad_right = min(im.shape[1] - xend, pad_right)
            xstart -= pad_left
            xend += pad_right
            pad_offset = pad_left - pad_right
            # print(pad_offset)
            if (h - w) % 2 == 1:
                pad_offset += 1
        else:
            pad_offset = 0

        c = im[ystart:yend, xstart:xend]
        if c.ndim != 3:
            assert c.ndim == 2
            c = np.tile(c[:, :, None], [1, 1, 3])

        # import matplotlib.pyplot as pt
        # pt.figure()
        # pt.imshow(c)
        # pt.show()


        image = c
        # print(image.shape)


        assert image.ndim == 3
        # assert image.shape[2] == 4, 'Only RGBA images are supported for now.'

        argmax = np.argmax(image.shape[:2]).item()

        smax = image.shape[argmax]
        smin = image.shape[1 - argmax]

        padding = (smax - smin) / 2
        before = math.floor(padding)
        after = math.ceil(padding)

        if argmax == 0:
            pad_width = [(0, 0), (before - pad_offset // 2, after + pad_offset // 2), (0, 0)]
        elif argmax == 1:
            pad_width = [(before - pad_offset // 2, after + pad_offset // 2), (0, 0), (0, 0)]
        else:
            raise RuntimeError()

        if fill_value is None:
            new_image = np.pad(image, pad_width, mode='edge')
        else:
            new_image = np.pad(image, pad_width, mode='constant', constant_values=fill_value)
        assert new_image.shape[0] == new_image.shape[1]
        # assert new_image.shape[2] == 4

        arr = new_image

        assert arr.shape[0] == arr.shape[1]
        arr = force_uint8(arr)
        resized = sp_misc.imresize(arr, size=(res, res), interp='lanczos')
        return force_float(resized)

    def fv(self):
        fv = io_utils.read_mesh(self.mesh_filename)
        R = pascal_to_shapenet_mesh_transformation()
        return transforms.apply44_mesh(R, fv)

    def cam_eye(self):
        # camera position after transforming to shapenet coordinates.
        inclination = 90 - self.elevation
        xyz = transforms.sph_to_xyz((1, inclination, self.azimuth - 90), is_input_radians=False)
        R = pascal_to_shapenet_mesh_transformation()
        xyz = transforms.apply44(R, xyz.reshape(1, 3)).ravel()
        return xyz

    def cam_up(self):
        up = np.array((0, 0, 1))  # up direction in pascal 3d

        R = pascal_to_shapenet_mesh_transformation()

        # transformed up vector. inverse transpose of R is R.
        up = transforms.apply44(R, up.reshape(1, 3)).ravel()

        # transformed position.
        xyz = self.cam_eye()

        # tilt
        roll = transforms.rotation_matrix(self.theta, -xyz / la.norm(xyz))
        up = transforms.apply44(roll, up.reshape(1, 3)).ravel()

        return up

    def cam_lookat(self):
        return np.array((0, 0, 0)).astype(np.float64)

    def cam_Rt(self):
        Rt = transforms.lookat_matrix(cam_xyz=self.cam_eye(), obj_xyz=self.cam_lookat(), up=self.cam_up())
        return Rt

    def cam_object(self):
        Rt = self.cam_Rt()
        cam = camera.OrthographicCamera.from_Rt(Rt, wh=(128, 128), is_world_to_cam=True)
        return cam
