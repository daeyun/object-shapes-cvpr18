import contextlib
import copy
import errno
import fcntl
import hashlib
import json
import numbers
import os
import sys
import tempfile
import time
import struct
from os import path

import numpy as np
import scipy.misc
import glob
import plyfile
import stl
import ujson
import blosc
import PIL.Image
import pyassimp

from dshin import log

# Specific to mvshape project
resources_dir = path.realpath(path.join(path.dirname(__file__), '../../resources'))


def read_mesh(filename):
    """
    :param filename: full path to mesh file.
    :return: dict with keys v and f.
        v: ndarray of size (num_vertices, 3), type uint32. zero-indexed.
        f: ndarray of size (num_faces, 3), type float64.
    """
    if filename.endswith('.off'):
        return read_off(filename)
    if filename.endswith('.ply'):
        return read_ply(filename)


def read_off_num_fv(filename: str) -> tuple:
    filename = path.expanduser(filename)
    with open(filename, 'r') as f:
        first_two_lines = [f.readline(), f.readline()]
    assert first_two_lines[0][:3] == 'OFF'

    tokens = ' '.join([item.strip() for item in [
        first_two_lines[0][3:],
        first_two_lines[1]
    ]]).split()

    num_vertices = int(tokens[0])
    num_faces = int(tokens[1])

    # TODO(daeyun): Make this a warning.
    # OK to fail.
    assert int(tokens[2]) == 0

    return num_faces, num_vertices


def read_off(filename):
    """
    Read OFF mesh files.

    File content must start with "OFF". Not always followed by a whitespace.
    """
    filename = path.expanduser(filename)
    with open(filename, 'r') as f:
        content = f.read()

    assert content[:3].upper() == 'OFF'
    content = content[4:] if content[3] == '\n' else content[3:]
    lines = content.splitlines()

    num_vertices, num_faces, _ = [int(val) for val in lines[0].split()]

    vertices = np.fromstring(' '.join(lines[1:num_vertices + 1]),
                             dtype=np.float64, sep=' ').reshape((-1, 3))
    faces = np.fromstring(
        ' '.join(lines[num_vertices + 1:num_vertices + num_faces + 1]),
        dtype=np.uint32,
        sep=' ').reshape((-1, 4))

    assert len(lines) == num_vertices + num_faces + 1
    assert (faces[:, 0] == 3).all(), "all triangle faces"

    faces = faces[:, 1:]

    if faces.min() != 0:
        print('faces.min() != 0', file=sys.stderr)

    if faces.max() != vertices.shape[0] - 1:
        print('faces.max() != vertices.shape[0]-1', file=sys.stderr)
        assert faces.max() < vertices.shape[0]

    assert vertices.shape[0] == num_vertices
    assert faces.shape[0] == num_faces

    return {
        'v': vertices,
        'f': faces,
    }


def save_stl(mesh, filename):
    faces = mesh['f']
    verts = mesh['v']
    stl_mesh = stl.mesh.Mesh(np.zeros(faces.shape[0], dtype=stl.mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = verts[f[j], :]
    stl_mesh.save(filename)


def read_ply(filename):
    plydata = plyfile.PlyData.read(filename)
    if 'vertex' in plydata and plydata['vertex']['x'].size > 0:
        v = np.vstack((plydata['vertex']['x'], plydata['vertex']['y'],
                       plydata['vertex']['z'])).T
    else:
        v = np.array([]).reshape([0, 3])

    if 'face' in plydata and plydata['face']['vertex_indices'].size > 0:
        inds = plydata['face']['vertex_indices']
        f = np.vstack(([i for i in inds]))
    else:
        f = np.array([]).reshape([0, 3])

    ret = {
        'v': v,
        'f': f,
    }

    return ret


def read_ply_pcl(filename):
    plydata = plyfile.PlyData.read(filename)
    v = np.vstack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'])).T
    normals = np.vstack((plydata['vertex']['nx'], plydata['vertex']['ny'], plydata['vertex']['nz'])).T
    values = plydata['vertex']['value'].T
    confidence = plydata['vertex']['confidence'].T
    ret = {
        'v': v,
        'confidence': confidence,
        'normals': normals,
        'value': values,
    }
    if 'face' in plydata:
        ret['f'] = plyfile.make2d(plydata['face'].data['vertex_indices'])
    return ret


def save_off(mesh, filename):
    verts = mesh['v'].astype(np.float32)
    faces = mesh['f'].astype(np.int32)

    filename = path.expanduser(filename)
    if not path.isdir(path.dirname(filename)):
        os.makedirs(path.dirname(filename))

    # Overwrite.
    if path.exists(filename):
        log.warn('Removing existing file %s', filename)
        os.remove(filename)

    with open(path.expanduser(filename), 'ab') as fp:
        fp.write('OFF\n{} {} 0\n'.format(
            verts.shape[0], faces.shape[0]).encode('utf-8'))
        np.savetxt(fp, verts, fmt='%.5f')
        np.savetxt(fp, np.hstack((3 * np.ones((
            faces.shape[0], 1)), faces)), fmt='%d')


def merge_meshes(*args):
    v = np.concatenate([item['v'] for item in args], axis=0)
    offsets = np.cumsum([0] + [item['v'].shape[0] for item in args])
    f = np.concatenate([item['f'] + offset for item, offset in zip(args, offsets)], axis=0)
    return {'v': v, 'f': f}


def sha1(objs):
    assert isinstance(objs, list), isinstance(objs, tuple)
    sha1 = hashlib.sha1()
    for obj in objs:
        sha1.update(str(obj).encode('utf8'))
    return sha1.hexdigest()


def sha256(objs):
    assert isinstance(objs, list), isinstance(objs, tuple)
    h = hashlib.sha256()
    for obj in objs:
        h.update(str(obj).encode('utf8'))
    return h.hexdigest()


def stringify_float_arrays(arr_list, precision=6):
    assert isinstance(arr_list, list), isinstance(arr_list, tuple)
    arr = np.hstack(arr_list).ravel().astype(np.float32)
    return np.array_str(arr, precision=precision, max_line_width=np.iinfo(np.int64).max)


def temp_filename(dirname='/tmp', prefix='', suffix=''):
    temp_name = next(tempfile._get_candidate_names())
    return path.join(dirname, prefix + temp_name + suffix)


def save_simple_points_ply(out_filename, points, text=False):
    assert points.shape[1] == 3
    assert points.ndim == 2

    dirname = path.dirname(out_filename)
    if not path.isdir(dirname):
        os.makedirs(dirname)
        log.info('mkdir -p {}'.format(dirname))

    xyz_nxyz = points
    vertex = np.core.records.fromarrays(xyz_nxyz.T, names='x, y, z',
                                        formats='f4, f4, f4')

    el = plyfile.PlyElement.describe(vertex, 'vertex')
    ply = plyfile.PlyData([el], text=text)
    ply.write(out_filename)

    # Replace \r\n with \n.
    with open(out_filename, 'rb') as f:
        content = f.read()
    beginning = content[:128]
    rest = content[128:]
    beginning = beginning.replace(b'ply\r\n', b'ply\n')

    with open(out_filename, 'wb') as f:
        f.write(beginning + rest)

    return ply


def save_points_ply(out_filename, points, normals, values, confidence, text=False, scale_normals_by_confidence=True):
    assert points.shape == normals.shape
    assert points.shape[1] == 3
    assert points.ndim == 2

    dirname = path.dirname(out_filename)
    if not path.isdir(dirname):
        os.makedirs(dirname)
        log.info('mkdir -p {}'.format(dirname))

    if values.ndim == 1:
        values = values[:, None]
    if confidence.ndim == 1:
        confidence = confidence[:, None]

    # Replace 0 confidence values with a small number to avoid zero-length normals.
    confidence = confidence.copy()
    confidence[np.isclose(confidence, 0.0)] = 1e-3

    if scale_normals_by_confidence:
        normals = normals.copy() * confidence
    xyz_nxyz = np.concatenate([points, normals, values, confidence], axis=1)
    vertex = np.core.records.fromarrays(xyz_nxyz.T, names='x, y, z, nx, ny, nz, value, confidence',
                                        formats='f4, f4, f4, f4, f4, f4, f4, f4')

    el = plyfile.PlyElement.describe(vertex, 'vertex')
    ply = plyfile.PlyData([el], text=text)
    ply.write(out_filename)

    # Replace \r\n with \n.
    with open(out_filename, 'rb') as f:
        content = f.read()
    beginning = content[:128]
    rest = content[128:]
    beginning = beginning.replace(b'ply\r\n', b'ply\n')

    with open(out_filename, 'wb') as f:
        f.write(beginning + rest)

    return ply


@contextlib.contextmanager
def open_locked(filename, mode='r+', sleep_seconds=0.2, verbose=True, **kwargs):
    """
    If `filename` does not exist, `mode` cannot be "r" or "r+".
    Parent directories are created if they do not exist.
    """
    dirname = path.dirname(filename)

    if 'r' in mode and not path.isfile(filename):
        raise FileNotFoundError('{} does not exist. `mode` cannot be {}.'.format(filename, mode))

    if not path.isdir(dirname):
        log.info('mkdir -p {}'.format(dirname))
        os.makedirs(dirname)

    with open(filename, mode, **kwargs) as fd:
        start_time = time.time()
        try:
            i = 0
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except IOError as e:
                    if e.errno != errno.EAGAIN:
                        raise e
                    else:
                        time.sleep(sleep_seconds)
                        if verbose and i > 0 and i % 10 == 0:
                            log.warning('Waiting for a locked file {}. Elapsed: {:.3g} seconds.'.format(filename,
                                                                                                        time.time() - start_time))
                        i += 1
            yield fd

            fd.flush()

        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)


class SimpleJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'dtype') and np.isscalar(obj):
            return obj.item()
        elif isinstance(obj, set):
            return tuple(obj)
        elif isinstance(obj, bytes):
            return obj.decode('utf-8')
        return json.JSONEncoder.default(self, obj)


@contextlib.contextmanager
def open_locked_json(filename):
    with open_locked(filename, mode='a+') as f:
        f.seek(0)
        try:
            content = f.read()
            data = json.loads(content)
        except (ValueError, FileNotFoundError):
            content = ''
            data = {}

        yield data

        try:
            new_content = json.dumps(data, cls=SimpleJSONEncoder)
        except Exception as ex:
            print(data)
            raise ex

        if content != new_content:
            f.seek(0)
            f.truncate()
            f.write(new_content)


@contextlib.contextmanager
def open_json_read_only(filename, fail_if_invalid=True):
    try:
        with open(filename, mode='r') as f:
            data = ujson.load(f)
    except (ValueError, FileNotFoundError) as ex:
        if fail_if_invalid:
            raise ex
        else:
            data = {}

    yield data


def update_json_file(filename, dict_values):
    assert isinstance(dict_values, dict)
    with open_locked_json(filename) as data:
        data.update(dict_values)


class DirectoryVersionManager(object):
    def __init__(self, filename='run_version.json', default_key='default'):
        assert isinstance(filename, str)
        assert isinstance(default_key, str)
        self._filename = filename
        self._default_key = default_key
        self._default_version = -1

    def _version_file(self, dirname) -> str:
        return path.join(path.realpath(path.expanduser(dirname)), self._filename)

    def dir_version(self, dirname, key=None, lock=True) -> int:
        if key is None:
            key = self._default_key
        assert isinstance(key, str)

        version_file = self._version_file(dirname)

        if not path.isfile(version_file):
            return self._default_version

        if lock:
            with open_locked_json(version_file) as dict_data:
                version_file_content = copy.deepcopy(dict_data)
        else:
            with open_json_read_only(version_file) as dict_data:
                version_file_content = copy.deepcopy(dict_data)

        if key not in version_file_content:
            return self._default_version

        version = version_file_content[key]

        if not (isinstance(version, numbers.Integral) or (isinstance(version, str) and version.isdigit())):
            log.warn('Invalid dir version found: {}'.format(version))
            return self._default_version
        version = int(version)
        return version

    def set_dir_version(self, dirname, version, key=None) -> int:
        assert isinstance(version, numbers.Integral) or (isinstance(version, str) and version.isdigit()), version
        if key is None:
            key = self._default_key
        assert isinstance(key, str)

        version_file = self._version_file(dirname)

        with open_locked_json(version_file) as data:
            if key in data:
                prev_version = data[key]
                if isinstance(prev_version, str) and prev_version.isdigit():
                    prev_version = int(prev_version)
                elif not isinstance(prev_version, numbers.Integral):
                    prev_version = self._default_version
            else:
                prev_version = self._default_version
            if prev_version > version:
                raise ValueError(
                    'Existing version {} is greater than {} in {}'.format(prev_version, version, version_file))
            data.update({key: version})

        return prev_version


def ensure_dir_exists(dirname, log_mkdir=True):
    dirname = path.realpath(path.expanduser(dirname))
    if not path.isdir(dirname):
        # `exist_ok` in case of race condition.
        os.makedirs(dirname, exist_ok=True)
        if log_mkdir:
            log.info('mkdir -p {}'.format(dirname))
    return dirname


def dir_child_basenames(dirpath):
    """
    Non recursively returns child directories and files in a directory.
    :param dirpath: Path of the directory. Must exist.
    :return: dirnames, filenames
    """
    dirpath = path.expanduser(dirpath)
    assert path.isdir(dirpath), '{} does not exist.'.format(dirpath)
    for _, dirnames, filenames in os.walk(dirpath):
        return dirnames, filenames
    raise AssertionError('This should not happen.')


def is_dir_empty(dirpath):
    dirnames, filenames = dir_child_basenames(dirpath)
    return len(dirnames) == 0 and len(filenames) == 0


def array_to_bytes(arr: np.ndarray):
    shape = arr.shape
    ndim = arr.ndim
    ret = struct.pack('i', ndim) + struct.pack('i' * ndim, *shape) + arr.tobytes(order='C')
    return ret


def bytes_to_array(s: bytes, dtype=np.float32):
    dims = struct.unpack('i', s[:4])[0]
    assert 0 <= dims < 1000
    shape = struct.unpack('i' * dims, s[4:4 * dims + 4])
    for dim in shape:
        assert dim > 0
    ret = np.frombuffer(s[4 * dims + 4:], dtype=dtype)
    assert ret.size == np.prod(shape), (ret.size, shape)
    ret.shape = shape
    return ret.copy()


def read_array(filename, dtype=np.float32):
    """
    Reads a binary file with the following format:
    [int32_t number of dimensions n]
    [int32_t dimension 0], [int32_t dimension 1], ..., [int32_t dimension n]
    [float data]

    :param filename: 
    :return:
    """
    with open(filename, mode='rb') as f:
        content = f.read()
    return bytes_to_array(content, dtype=dtype)


def read_array_compressed(filename, dtype=np.float32):
    """
    Reads a binary file compressed with Blosc. Otherwise the same as read_float32_array.
    """
    with open(filename, mode='rb') as f:
        compressed = f.read()
    decompressed = blosc.decompress(compressed)
    return bytes_to_array(decompressed, dtype=dtype)


def save_array_compressed(filename, arr: np.ndarray):
    encoded = array_to_bytes(arr)
    compressed = blosc.compress(encoded, arr.dtype.itemsize, clevel=9, shuffle=True, cname='lz4hc')
    with open(filename, mode='wb') as f:
        f.write(compressed)


def read_png(filename):
    """
    :param filename:
    :return: ndarray of shape (height, width, channels). 0 to 255.
    the last channel is alpha. Also 0 to 255.
    """
    image = scipy.misc.imread(filename)

    if image.dtype != np.uint8:
        raise NotImplementedError('Unrecognized image format: {}'.format(image.dtype))

    return image


def read_jpg(filename):
    """
    :param filename:
    :return: ndarray of shape (height, width, channels). 0 to 255.
    the last channel is alpha. Also 0 to 255.
    """
    image = scipy.misc.imread(filename)

    if image.dtype != np.uint8:
        raise NotImplementedError('Unrecognized image format: {}'.format(image.dtype))

    return image


def make_resources_path(p: str):
    if p.startswith('/'):
        p = p[1:]
    assert len(p) > 0
    ret = path.join(resources_dir, p)
    return ret


def save_png(image: np.ndarray, filename: str):
    assert image.ndim == 3
    assert image.shape[2] in (3, 4)
    assert filename.endswith('.png'), 'failed filename sanity check: {}'.format(filename)
    assert image.dtype == np.uint8
    im = PIL.Image.fromarray(image)
    im.save(filename, optimize=True)


def read_obj(filename):
    scene = pyassimp.load(filename)

    vertices = []
    faces = []
    vertex_count = 0
    for m in scene.meshes:
        f = m.faces
        v = m.vertices
        faces.append(f + vertex_count)
        vertices.append(v)
        num_vertices = v.shape[0]
        vertex_count += num_vertices
    faces = np.concatenate(faces, axis=0)
    vertices = np.concatenate(vertices, axis=0)

    fv = {'v': vertices, 'f': faces}

    pyassimp.release(scene)

    return fv
