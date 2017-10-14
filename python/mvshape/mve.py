import array
import configparser
import enum
import glob
import subprocess
import textwrap
import time
import typing
from os import path

import numpy as np
import plyfile
from scipy import misc as sp_misc

from dshin import camera
from dshin import log
from dshin import io_utils


class MVEImageType(enum.Enum):
    IMAGE_TYPE_UNKNOWN = 0
    IMAGE_TYPE_UINT8 = 1
    IMAGE_TYPE_UINT16 = 2
    IMAGE_TYPE_UINT32 = 3
    IMAGE_TYPE_UINT64 = 4
    IMAGE_TYPE_SINT8 = 5
    IMAGE_TYPE_SINT16 = 6
    IMAGE_TYPE_SINT32 = 7
    IMAGE_TYPE_SINT64 = 8
    IMAGE_TYPE_FLOAT = 9
    IMAGE_TYPE_DOUBLE = 10


_np_to_mve_image_types = {
    np.uint8: MVEImageType.IMAGE_TYPE_UINT8,
    np.uint16: MVEImageType.IMAGE_TYPE_UINT16,
    np.uint32: MVEImageType.IMAGE_TYPE_UINT32,
    np.uint64: MVEImageType.IMAGE_TYPE_UINT64,
    np.int8: MVEImageType.IMAGE_TYPE_SINT8,
    np.int16: MVEImageType.IMAGE_TYPE_SINT16,
    np.int32: MVEImageType.IMAGE_TYPE_SINT32,
    np.float32: MVEImageType.IMAGE_TYPE_FLOAT,
    np.float64: MVEImageType.IMAGE_TYPE_DOUBLE,
}

_mve_to_np_image_types = {v.value: k for k, v in _np_to_mve_image_types.items()}

_header = array.array('B', [0x89, 0x4D, 0x56, 0x45, 0x5F, 0x49, 0x4D, 0x41, 0x47, 0x45, 0x0A])

_cpp_binary_root = path.realpath(path.join(path.dirname(path.realpath(__file__)), '../../cmake-build-release/'))
_mve_root = path.realpath(path.join(path.dirname(path.realpath(__file__)), '../../third_party/repos/mve'))


def write_mvei_image(filename, image):
    """
    Writes a numpy array image as .mvei file.
    See https://github.com/simonfuhrmann/mve/wiki/MVE-File-Format

    :param filename: Output filename. Must end with `.mvei`.
    :param image: An array of shape (height, width) or (height, width, channel). Channel must be one of 1, 3, 4.
    """
    assert filename.endswith('.mvei')
    assert image.ndim in (2, 3)
    dirname = path.dirname(filename)
    io_utils.ensure_dir_exists(dirname)
    with open(filename, 'wb') as f:
        _header.tofile(f)
        if image.ndim == 2:
            w, h, c = image.shape[1], image.shape[0], 1
        else:
            w, h, c = image.shape[2], image.shape[1], image.shape[0]
        assert c in (1, 3, 4)
        t = _np_to_mve_image_types[image.dtype.type].value
        arr = array.array('i', [w, h, c, t])
        arr.tofile(f)
        image.tofile(f)


def read_mvei_image(filename):
    """
    Reads `.mvei` image as a numpy array.
    See https://github.com/simonfuhrmann/mve/wiki/MVE-File-Format

    :param filename: Input .mvei filename.
    :return: An array of shape (height, width, channel).
    """
    with open(filename, 'rb') as f:
        data = f.read()
        header_size = array.array('B').itemsize * len(_header)
        metadata_size = array.array('i').itemsize * 4

        header = array.array('B')
        header.frombytes(data[:header_size])
        assert header == _header

        whct = array.array('i')
        whct.frombytes(data[header_size:header_size + metadata_size])

        image_data = array.array('f')
        image_data.frombytes(data[header_size + metadata_size:])
        image = np.frombuffer(image_data, dtype=_mve_to_np_image_types[whct[3]])
        image.shape = (whct[1], whct[0], whct[2])
    return image


def save_as_mve_views(scene_dir, masked_depth_images: np.ndarray, ortho_cameras: typing.Sequence[camera.OrthographicCamera], cam_scale=1.0):
    """
    Write depth images as MVE views to be used as input. Currently only supports orthographic cameras.

    :param scene_dir: Output files are saved in a subdirectory called `views`. Created if not exists.
    :param depth_images: An array of shape (num_views, height, width).
    :param ortho_cameras: Orthographic cameras.
    :param cam_scale: Scale factor after camera transformation.
    :return:
    """
    views_dir = path.join(scene_dir, 'views')
    io_utils.ensure_dir_exists(views_dir)
    assert masked_depth_images.ndim == 3

    depth_images = masked_depth_images.copy()
    depth_images[np.isnan(depth_images)] = 0

    for i in range(len(masked_depth_images)):
        masked_depth_image = masked_depth_images[i]
        unscaled_masked_depth_image = masked_depth_image / cam_scale
        depth_image = depth_images[i]
        view_dirname = path.join(views_dir, 'view_{0:04d}.mve'.format(i))
        write_mvei_image(path.join(view_dirname, 'depth-L0.mvei'), unscaled_masked_depth_image)
        sp_misc.imsave(path.join(view_dirname, 'original.png'), depth_image)

        config = configparser.ConfigParser()
        config['view'] = {'id': i, 'name': i}

        cam = ortho_cameras[i]

        t = np.copy(cam.t)
        R = np.copy(cam.R)

        config['camera'] = {
            'rotation': ' '.join([repr(float(item)) for item in R.ravel().tolist()]),
            'translation': ' '.join([repr(float(item)) for item in t.ravel().tolist()]),
            'focal_length': '-1',
            'pixel_aspect': '-1',
            'principal_point': '0 0',
        }

        config['orthographic_camera'] = {
            'top': cam.trbl[0] / cam_scale,
            'right': cam.trbl[1] / cam_scale,
            'bottom': cam.trbl[2] / cam_scale,
            'left': cam.trbl[3] / cam_scale,
        }

        with open(path.join(view_dirname, 'meta.ini'), 'w') as f:
            config.write(f)


def run_command(command):
    log.debug(subprocess.list2cmdline(command))

    start_time = time.time()
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return_code = p.wait()
    log.debug('return code: {}'.format(return_code))
    elapsed = time.time() - start_time

    stdout, stderr = p.communicate()
    stdout = stdout.decode('utf-8')
    stderr = stderr.decode('utf-8')

    if p.returncode != 0:
        exec_summary = textwrap.dedent(textwrap.dedent("""
        Command: {}
        Return code: {}

        stdout:
        {}

        stderr:
        {}
        """).format(
            subprocess.list2cmdline(command),
            p.returncode,
            stdout,
            stderr
        ))
        raise RuntimeError(exec_summary)

    log.info('{0} took {1:.3f} seconds.'.format(path.basename(command[0]), elapsed))

    return p, stdout, stderr


def convert_mve_views_to_meshes(scene_dir):
    executable = path.join(_cpp_binary_root, 'cpp/apps/depth2mesh')
    assert path.isfile(executable), '{} does not exist. Run build.sh.'.format(executable)

    depth_mesh_dir = path.join(scene_dir, 'depth_meshes')
    io_utils.ensure_dir_exists(depth_mesh_dir)

    command = [executable, scene_dir, path.join(depth_mesh_dir, 'mesh.ply')]
    assert path.isdir(scene_dir)
    run_command(command)

    return depth_mesh_dir


def fssr_from_scene_dir(scene_dir, scale=1):
    executable = path.join(_mve_root, 'apps/fssrecon/fssrecon')
    assert path.isfile(executable), '{} does not exist. Run build.sh.'.format(executable)

    depth_mesh_dir = path.join(scene_dir, 'depth_meshes')
    assert path.isdir(depth_mesh_dir)

    fssr_dir = path.join(scene_dir, 'fssr')
    io_utils.ensure_dir_exists(fssr_dir)

    input_files = glob.glob(path.join(depth_mesh_dir, '*.ply'))
    output_file = path.join(fssr_dir, 'recon.ply')
    assert path.isfile(input_files[0])
    command = [executable, '--scale-factor={0:.5f}'.format(scale), '--refine-octree=0'] + input_files + [output_file]
    run_command(command)

    return output_file


# newer version
def fssr_pcl_files(pcl_ply_files, scale=1):
    executable = path.join(_mve_root, 'apps/fssrecon/fssrecon')
    assert path.isfile(executable), '{} does not exist. Run build.sh.'.format(executable)

    input_files = []
    for item in pcl_ply_files:
        assert path.isfile(item)
        input_files.append(item)

    assert len(input_files) > 0

    prefix, ext = path.splitext(pcl_ply_files[0])
    output_file = path.join(path.dirname(prefix), 'fssr_recon' + ext)

    command = [executable, '--scale-factor={0:.5f}'.format(scale), '--refine-octree=0'] + input_files + [output_file]
    run_command(command)

    assert path.isfile(output_file), output_file

    return output_file


# deprecated
def fssr_pcl(pcl_ply_file, scale=1):
    executable = path.join(_mve_root, 'apps/fssrecon/fssrecon')
    assert path.isfile(executable), '{} does not exist. Run build.sh.'.format(executable)

    prefix, ext = path.splitext(pcl_ply_file)
    output_file = prefix + '_fssr_recon' + ext

    assert path.isfile(pcl_ply_file)
    command = [executable, '--scale-factor={0:.5f}'.format(scale), '--refine-octree=0', pcl_ply_file, output_file]
    run_command(command)

    return output_file


def meshclean(meshfile, threshold=1.0):
    executable = path.join(_mve_root, 'apps/meshclean/meshclean')
    assert path.isfile(executable), '{} does not exist. Run build.sh.'.format(executable)

    assert path.isfile(meshfile)

    prefix, ext = path.splitext(meshfile)
    outfile = prefix + '_clean' + ext

    # TODO(daeyun): Adjust parameters.

    command = [executable, '--threshold={0:.5f}'.format(threshold), meshfile, outfile]
    assert path.isfile(meshfile)
    run_command(command)

    assert path.isfile(outfile), outfile
    return outfile


def merge_ply(mesh_filenames, out_filename):
    plydata_list = []
    for mesh_file in mesh_filenames:
        plydata = plyfile.PlyData.read(mesh_file)
        plydata_list.append(plydata)

    merged_vertices = []
    merged_faces = []
    index_offset = 0
    has_face = False
    for plydata in plydata_list:
        property_names = [el.name for el in plydata.elements]

        assert len(property_names) <= 2
        assert 'vertex' in property_names
        if len(property_names) == 2:
            assert 'face' in property_names
        vdata = plydata['vertex'].data
        merged_vertices.append(vdata)

        if 'face' in property_names:
            has_face = True
            plydata['face'].data['vertex_indices'] += index_offset
            merged_faces.append(plydata['face'].data)

        index_offset += len(vdata)
    merged_vertices = np.concatenate(merged_vertices, axis=0)
    merged_faces = np.concatenate(merged_faces, axis=0)

    plydata = plydata_list[0]
    plydata['vertex'].data = merged_vertices
    if has_face:
        plydata['face'].data = merged_faces

    plydata.write(out_filename)
