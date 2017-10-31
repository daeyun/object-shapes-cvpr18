import numpy as np
import peewee
import hashlib
from playhouse.sqlite_ext import SqliteExtDatabase

db = SqliteExtDatabase(None, threadlocals=True)


class Vec3Field(peewee.Field):
    db_field = 'vec3'

    def db_value(self, value: np.ndarray):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            value = np.array(value, dtype=np.float32)
        if value.dtype != np.float32:
            value = value.astype(np.float32)
        if value.ndim != 1:
            value = value.ravel()
        assert isinstance(value, np.ndarray)
        assert value.size == 3
        assert value.ndim == 1
        assert value.dtype == np.float32
        return value.tostring()

    def python_value(self, value):
        if value is None:
            return None
        return np.fromstring(value, dtype=np.float32)


class TensorShapeField(peewee.Field):
    db_field = 'tensor_shape'

    def db_value(self, value: np.ndarray):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            value = np.array(value, dtype=np.int32)
        if value.dtype != np.int32:
            value = value.astype(np.int32)
        if value.ndim != 1:
            value = value.ravel()
        assert isinstance(value, np.ndarray)
        # value.size could be 0 (i.e. empty array) if the tensor is a scalar.
        assert value.ndim == 1
        assert value.dtype == np.int32
        return value.tostring()

    def python_value(self, value):
        if value is None:
            return None
        return np.fromstring(value, dtype=np.int32)


class BaseModel(peewee.Model):
    class Meta:
        database = db


class Example(BaseModel):
    pass


class Tag(BaseModel):
    name = peewee.CharField(unique=True, max_length=128, index=True)
    description = peewee.TextField(null=True)


class ExampleTag(BaseModel):
    """ Many-to-many. """
    example = peewee.ForeignKeyField(Example)
    tag = peewee.ForeignKeyField(Tag)


class Dataset(BaseModel):
    name = peewee.CharField(unique=True, max_length=128, index=True)
    description = peewee.TextField(null=True)


class ExampleDataset(BaseModel):
    """ Many-to-many. """
    example = peewee.ForeignKeyField(Example)
    dataset = peewee.ForeignKeyField(Dataset)


class Split(BaseModel):
    """
    Train, test, validation, validation_subset, etc..
    """
    name = peewee.CharField(max_length=128)


class ExampleSplit(BaseModel):
    """ Many-to-many. """
    example = peewee.ForeignKeyField(Example)
    split = peewee.ForeignKeyField(Split)


class Category(BaseModel):
    name = peewee.CharField(unique=True, max_length=128, index=True)


class Object(BaseModel):
    """
    Usually a ground truth mesh if it is a synthetic object.
    """
    # Name does not have to be unique, but filename has to be.
    # There are cases where an object has the same name but different paths.
    # In that case, create_or_get will just make a new one.
    name = peewee.CharField(null=True, max_length=128, index=True)
    category = peewee.ForeignKeyField(Category)
    # This can be null.
    dataset = peewee.ForeignKeyField(Dataset, null=True)
    num_vertices = peewee.IntegerField(null=True)
    num_faces = peewee.IntegerField(null=True)
    mesh_filename = peewee.TextField(null=True, unique=True)


class Camera(BaseModel):
    position_xyz = Vec3Field()
    up = Vec3Field()
    lookat = Vec3Field()
    is_orthographic = peewee.BooleanField(default=True)
    fov = peewee.FloatField(null=True)
    scale = peewee.FloatField(null=True)  # Scale of the frustum's top, left, bottom, right parameters.


class RenderingType(BaseModel):
    """
    rgb, normal, depth, voxels
    """
    name = peewee.CharField(unique=True, max_length=128, index=True)


class ObjectRendering(BaseModel):
    """
    Shape representation of an object with a reference camera.
    """
    type = peewee.ForeignKeyField(RenderingType)
    camera = peewee.ForeignKeyField(Camera)
    object = peewee.ForeignKeyField(Object)
    filename = peewee.TextField(unique=True)
    resolution = peewee.IntegerField(null=True, index=True)
    num_channels = peewee.FixedCharField(null=True, index=True)
    set_size = peewee.FixedCharField(default=1, index=True)
    is_normalized = peewee.BooleanField(default=False, index=True)


class ExampleObjectRendering(BaseModel):
    """ Many-to-many. """
    example = peewee.ForeignKeyField(Example)
    rendering = peewee.ForeignKeyField(ObjectRendering)


def stringify_float_arrays(arr_list, precision=6):
    assert isinstance(arr_list, (list, tuple))
    arr = np.hstack(arr_list).ravel().astype(np.float32)
    return np.array_str(arr, precision=precision, max_line_width=np.iinfo(np.int64).max)


def sha256(objs):
    assert isinstance(objs, list), isinstance(objs, tuple)
    h = hashlib.sha256()
    for obj in objs:
        h.update(str(obj).encode('utf8'))
    return h.hexdigest()


def camera_hash(camera: Camera) -> str:
    """
    64 hexadecimal characters.
    """
    if camera.is_orthographic:
        string = stringify_float_arrays(
            [camera.position_xyz, camera.up, camera.lookat, ],
            precision=6)
    else:
        string = stringify_float_arrays(
            [camera.position_xyz, camera.up, camera.lookat, camera.fov, ],
            precision=6)
    return sha256([string])


@db.aggregate('concat_str', 1)
class ConcatStrings(object):
    def __init__(self):
        self.strings = []

    def step(self, value):
        self.strings.append(value)

    def finalize(self):
        return ','.join(sorted(list(set(self.strings))))


def init(sqlite_path):
    """
    Connects and creates all tables that do not already exist.

    :param sqlite_path: Path to sqlite database file. A new one is created if one does not exist.
    :return: The singleton `peewee.Database` object. Likely not needed for anything.
    """
    # autocommit=True by default.

    # TODO(daeyun): find and pick shards.
    # prefix, ext = path.splitext(sqlite_path)
    # glob.glob(prefix + '.shard*' + ext)

    db.init(sqlite_path)
    db.connect()

    db.create_tables(
        [Tag, ExampleTag, Dataset, ExampleDataset, Split, ExampleSplit, Category, Object, Camera,
         RenderingType, ObjectRendering, ExampleObjectRendering, Example], safe=True)
    return db
