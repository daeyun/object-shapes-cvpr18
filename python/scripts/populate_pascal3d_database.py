import shutil
import os
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
from mvshape import pascal
from os.path import join
from dshin import log
import json

base = '/data/mvshape'


def make_dbpath(fullpath):
    return re.match(r'.*?(/pascal3d/.*$)', fullpath).group(1)


def make_fullpath(dbpath):
    assert dbpath.startswith('/')
    ret = base + dbpath
    return ret


tags = {}
splits = {}
rendering_types = {}
datasets = {}
categories = {}
db_object_map = {}
db_object_centered_renderings = {}
_cached_db_cameras = {}
output_cam_distance_from_origin = 2
annos = {}


def make_tag(name):
    tags[name] = dbm.Tag.get_or_create(name=name)[0]


def make_split(name):
    splits[name] = dbm.Split.get_or_create(name=name)[0]


def make_rendering_type(name):
    rendering_types[name] = dbm.RenderingType.get_or_create(name=name)[0]


def make_dataset(name):
    datasets[name] = dbm.Dataset.get_or_create(name=name)[0]


def get_db_camera(cam: camera.OrthographicCamera, fov=None) -> dbm.Camera:
    sRt_hash = cam.sRt_hash()
    if fov is not None:
        assert fov > 0
        fov = float(fov)
        sRt_hash += '_{:.4f}'.format(fov)

    if sRt_hash not in _cached_db_cameras:
        if fov is None:
            db_cam = dbm.Camera(
                position_xyz=cam.pos,
                up=cam.up_vector,
                lookat=np.array((0, 0, 0), dtype=np.float64),
                is_orthographic=True,
                scale=1.75,
            )
        else:
            db_cam = dbm.Camera(
                position_xyz=cam.pos,
                up=cam.up_vector,
                lookat=np.array((0, 0, 0), dtype=np.float64),
                is_orthographic=False,
                fov=fov,
                scale=1.0,
            )
        db_cam.save()

        _cached_db_cameras[sRt_hash] = db_cam

    return _cached_db_cameras[sRt_hash]


def load_annotation_object(matfile):
    if matfile not in annos:
        anno = pascal.PascalAnnotation(make_fullpath(matfile))
        annos[matfile] = anno
    return annos[matfile]


def main():
    with open('/data/mvshape/pascal3d/eval_1500.json', 'r') as f:
        eval_1500 = json.load(f)

    with open('/data/mvshape/pascal3d/validation_1200.json', 'r') as f:
        validation_1200 = json.load(f)

    validation_matfiles = []
    for c, l in validation_1200.items():
        for fname in l:
            validation_matfiles.append(fname)
    test_matfiles = []
    for c, l in eval_1500.items():
        for fname in l:
            test_matfiles.append(fname)

    database_filename = '/data/mvshape/database/pascal3d.sqlite'
    if path.isfile(database_filename):
        os.remove(database_filename)
    dbm.init(database_filename)

    with dbm.db.transaction() as txn:
        make_dataset('pascal3d')
        make_rendering_type('rgb')
        make_rendering_type('depth')
        make_rendering_type('voxels')
        make_rendering_type('normal')
        make_tag('novelview')
        make_tag('novelmodel')
        make_tag('novelclass')
        make_tag('perspective_input')
        make_tag('orthographic_input')
        make_tag('perspective_output')
        make_tag('orthographic_output')
        make_tag('viewer_centered')
        make_tag('object_centered')
        make_tag('real_world')
        make_split('test')
        make_split('validation')

        # Quote from http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1/README.txt
        #   "The OBJ files have been pre-aligned so that the up direction is the +Y axis, and the front is the +X axis.  In addition each model is normalized to fit within a unit cube centered at the origin."
        oc_output_cam = camera.OrthographicCamera.from_Rt(transforms.lookat_matrix(cam_xyz=(0, 0, output_cam_distance_from_origin),
                                                                                   obj_xyz=(0, 0, 0),
                                                                                   up=(0, 1, 0)),
                                                          wh=(128, 128),
                                                          is_world_to_cam=True)
        db_oc_output_cam = get_db_camera(oc_output_cam, fov=None)
        txn.commit()

        all_matfiles = test_matfiles + validation_matfiles

        # populate categories
        for matfile in all_matfiles:
            anno = load_annotation_object(matfile)
            if anno.category_name in categories:
                continue
            db_category_i, _ = dbm.Category.get_or_create(name=anno.category_name)
            categories[anno.category_name] = db_category_i

        for matfile in all_matfiles:
            anno = load_annotation_object(matfile)
            model_name = '{}_{:02d}'.format(anno.category_name, anno.cad_index)
            if model_name in db_object_map:
                continue

            new_mesh_filename_db = '/mesh/pascal3d/{}.off'.format(model_name)
            new_mesh_filename = make_fullpath(new_mesh_filename_db)

            if not path.isfile(new_mesh_filename):
                io_utils.save_off(anno.fv(), new_mesh_filename)

            db_object = dbm.Object.create(
                name=model_name,
                category=categories[anno.category_name],
                dataset=datasets['pascal3d'],
                num_vertices=0,  # Not needed for now. Easy to fill in later.
                num_faces=0,
                mesh_filename=new_mesh_filename_db,
            )
            db_object_map[model_name] = db_object

            oc_rendering_name = model_name

            assert model_name not in db_object_centered_renderings
            db_object_centered_renderings[model_name] = {
                'output_rgb': dbm.ObjectRendering.create(
                    type=rendering_types['rgb'],
                    camera=db_oc_output_cam,
                    object=db_object,
                    # JPG
                    filename='/pascal3d/mv6_rgb_128/{}.bin'.format(oc_rendering_name),
                    resolution=128,
                    num_channels=3,
                    set_size=6,
                    is_normalized=False,
                ),
                'output_depth': dbm.ObjectRendering.create(
                    type=rendering_types['depth'],
                    camera=db_oc_output_cam,
                    object=db_object,
                    # Since there is only one gt rendering per model, their id is the same as the model name.
                    filename='/pascal3d/mv6_depth_128/{}.bin'.format(oc_rendering_name),
                    resolution=128,
                    num_channels=1,
                    set_size=6,
                    is_normalized=False,
                ),
                'output_normal': dbm.ObjectRendering.create(
                    type=rendering_types['normal'],
                    camera=db_oc_output_cam,
                    object=db_object,
                    filename='/pascal3d/mv6_normal_128/{}.bin'.format(oc_rendering_name),
                    resolution=128,
                    num_channels=3,
                    set_size=6,
                    is_normalized=False,
                ),
                'output_voxels': dbm.ObjectRendering.create(
                    type=rendering_types['voxels'],
                    camera=db_oc_output_cam,
                    object=db_object,
                    filename='/pascal3d/voxels_32/{}.bin'.format(oc_rendering_name),
                    resolution=32,
                    num_channels=1,
                    set_size=1,
                    is_normalized=False,
                )
            }

        txn.commit()

        for matfile in all_matfiles:
            anno = load_annotation_object(matfile)  # type: pascal.PascalAnnotation
            model_name = '{}_{:02d}'.format(anno.category_name, anno.cad_index)
            input_cam = anno.cam_object()
            # 49.1343 degrees is the default fov in blender.
            db_input_cam = get_db_camera(input_cam, fov=49.1343)

            input_cam_depth_xyz = (input_cam.pos / la.norm(input_cam.pos)) * 1.5
            input_cam_depth_Rt = transforms.lookat_matrix(cam_xyz=input_cam_depth_xyz,
                                                          obj_xyz=(0, 0, 0),
                                                          up=input_cam.up_vector)
            input_cam_depth = camera.OrthographicCamera.from_Rt(input_cam_depth_Rt, wh=(128, 128), is_world_to_cam=True)
            db_input_cam_depth = get_db_camera(input_cam_depth, fov=49.1343)

            output_cam_xyz = (input_cam.pos / la.norm(input_cam.pos)) * output_cam_distance_from_origin
            output_Rt = transforms.lookat_matrix(cam_xyz=output_cam_xyz,
                                                 obj_xyz=(0, 0, 0),
                                                 up=input_cam.up_vector)
            vc_output_cam = camera.OrthographicCamera.from_Rt(output_Rt, wh=(128, 128), is_world_to_cam=True)
            db_vc_output_cam = get_db_camera(vc_output_cam, fov=None)

            # ---
            db_object = db_object_map[model_name]

            # e.g. aeroplane_01_n03693474_7232
            vc_rendering_name = '{}_{}'.format(model_name, anno.jpg_filename.split('.')[0])

            # Input rgb image:
            db_object_rendering_input_rgb = dbm.ObjectRendering.create(
                type=rendering_types['rgb'],
                camera=db_input_cam,
                object=db_object,
                # This should already exist.
                filename='/pascal3d/single_rgb_128/{}.png'.format(vc_rendering_name),
                resolution=128,
                num_channels=1,
                set_size=1,
                is_normalized=False,  # False for rgb.
            )

            db_object_rendering_input_depth = dbm.ObjectRendering.create(
                type=rendering_types['depth'],
                camera=db_input_cam_depth,
                object=db_object,
                filename='/pascal3d/single_depth_128/{}.bin'.format(vc_rendering_name),
                resolution=128,
                num_channels=1,
                set_size=1,
                is_normalized=True,
            )

            db_object_rendering_vc_output_rgb = dbm.ObjectRendering.create(
                type=rendering_types['rgb'],
                camera=db_vc_output_cam,
                object=db_object,
                filename='/pascal3d/mv6_rgb_128/{}.bin'.format(vc_rendering_name),
                resolution=128,
                num_channels=3,
                set_size=6,
                is_normalized=False,
            )

            db_object_rendering_vc_output_depth = dbm.ObjectRendering.create(
                type=rendering_types['depth'],
                camera=db_vc_output_cam,
                object=db_object,
                filename='/pascal3d/mv6_depth_128/{}.bin'.format(vc_rendering_name),
                resolution=128,
                num_channels=1,
                set_size=6,
                is_normalized=False,
            )

            db_object_rendering_vc_output_normal = dbm.ObjectRendering.create(
                type=rendering_types['normal'],
                camera=db_vc_output_cam,
                object=db_object,
                filename='/pascal3d/mv6_normal_128/{}.bin'.format(vc_rendering_name),
                resolution=128,
                num_channels=3,
                set_size=6,
                is_normalized=False,
            )

            db_object_rendering_vc_output_voxels = dbm.ObjectRendering.create(
                type=rendering_types['voxels'],
                camera=db_vc_output_cam,
                object=db_object,
                filename='/pascal3d/voxels_32/{}.bin'.format(vc_rendering_name),
                resolution=32,
                num_channels=1,
                set_size=1,
                is_normalized=False,
            )

            # Examples
            # ----------------

            is_test = matfile in test_matfiles
            is_validation = matfile in validation_matfiles
            assert is_test ^ is_validation

            if is_test:
                split = splits['test']
            elif is_validation:
                split = splits['validation']
            else:
                raise NotImplementedError()

            # A row in the `Example` table is just an id for many-to-many references.

            # View centered
            example_viewer_centered = dbm.Example.create()
            dbm.ExampleObjectRendering.create(example=example_viewer_centered, rendering=db_object_rendering_input_rgb)
            dbm.ExampleObjectRendering.create(example=example_viewer_centered, rendering=db_object_rendering_input_depth)
            dbm.ExampleObjectRendering.create(example=example_viewer_centered, rendering=db_object_rendering_vc_output_depth)
            dbm.ExampleObjectRendering.create(example=example_viewer_centered, rendering=db_object_rendering_vc_output_normal)
            dbm.ExampleObjectRendering.create(example=example_viewer_centered, rendering=db_object_rendering_vc_output_rgb)
            dbm.ExampleObjectRendering.create(example=example_viewer_centered, rendering=db_object_rendering_vc_output_voxels)
            dbm.ExampleDataset.create(example=example_viewer_centered, dataset=datasets['pascal3d'])
            dbm.ExampleSplit.create(example=example_viewer_centered, split=split)
            dbm.ExampleTag.create(example=example_viewer_centered, tag=tags['real_world'])
            dbm.ExampleTag.create(example=example_viewer_centered, tag=tags['viewer_centered'])
            dbm.ExampleTag.create(example=example_viewer_centered, tag=tags['perspective_input'])
            dbm.ExampleTag.create(example=example_viewer_centered, tag=tags['orthographic_output'])
            dbm.ExampleTag.create(example=example_viewer_centered, tag=tags['novelmodel'])

            # Object centered
            example_object_centered = dbm.Example.create()
            dbm.ExampleObjectRendering.create(example=example_object_centered, rendering=db_object_rendering_input_rgb)
            dbm.ExampleObjectRendering.create(example=example_object_centered, rendering=db_object_rendering_input_depth)
            dbm.ExampleObjectRendering.create(example=example_object_centered, rendering=db_object_centered_renderings[model_name]['output_depth'])
            dbm.ExampleObjectRendering.create(example=example_object_centered, rendering=db_object_centered_renderings[model_name]['output_normal'])
            dbm.ExampleObjectRendering.create(example=example_object_centered, rendering=db_object_centered_renderings[model_name]['output_rgb'])
            dbm.ExampleObjectRendering.create(example=example_object_centered, rendering=db_object_centered_renderings[model_name]['output_voxels'])
            dbm.ExampleDataset.create(example=example_object_centered, dataset=datasets['pascal3d'])
            dbm.ExampleSplit.create(example=example_object_centered, split=split)
            dbm.ExampleTag.create(example=example_object_centered, tag=tags['real_world'])
            dbm.ExampleTag.create(example=example_object_centered, tag=tags['object_centered'])
            dbm.ExampleTag.create(example=example_object_centered, tag=tags['perspective_input'])
            dbm.ExampleTag.create(example=example_object_centered, tag=tags['orthographic_output'])
            dbm.ExampleTag.create(example=example_object_centered, tag=tags['novelmodel'])

        txn.commit()

    dbm.db.commit()


if __name__ == '__main__':
    main()
