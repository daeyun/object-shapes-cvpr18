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
from os.path import join
from dshin import log


def main2():
    db_filename = '/data/mvshape/database/shapenet.sqlite'
    # shutil.copy(db_filename)

    # glob.glob('/data/render_for_cnn/data/syn_images_cropped_bkg_overlaid/02691156/10155655850468db78d106ce0a280f87/')

    params_file = '/data/render_for_cnn/data/syn_images/02691156/1a04e3eab45ca15dd86060f189eb133/params.txt'
    with open(params_file, 'r') as f:
        content = f.read()
    lines = [[float(item) for item in re.split(r'\s+', line.strip())]
             for line in re.split(r'\r?\n', content) if line]

    cameras = []

    for params in lines:
        Rt = render_for_cnn_utils.get_Rt_from_RenderForCNN_parameters(params)
        cam = camera.OrthographicCamera.from_Rt(Rt=Rt, wh=(128, 128), is_world_to_cam=True)
        cameras.append(cam)

    meshfile = '/data/shapenetcore/ShapeNetCore.v1/02691156/1a04e3eab45ca15dd86060f189eb133/model.obj'


_cached_db_cameras = {}


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


tags = {}
splits = {}
rendering_types = {}
datasets = {}


def make_tag(name):
    tags[name] = dbm.Tag.get_or_create(name=name)[0]


def make_split(name):
    splits[name] = dbm.Split.get_or_create(name=name)[0]


def make_rendering_type(name):
    rendering_types[name] = dbm.RenderingType.get_or_create(name=name)[0]


def make_dataset(name):
    datasets[name] = dbm.Dataset.get_or_create(name=name)[0]


synset_name_pairs = [('02691156', 'aeroplane'),
                     ('02834778', 'bicycle'),
                     ('02858304', 'boat'),
                     ('02876657', 'bottle'),
                     ('02924116', 'bus'),
                     ('02958343', 'car'),
                     ('03001627', 'chair'),
                     ('04379243', 'diningtable'),
                     ('03790512', 'motorbike'),
                     ('04256520', 'sofa'),
                     ('04468005', 'train'),
                     ('03211117', 'tvmonitor')]
synset_name_map = dict(synset_name_pairs)

is_subset = True


def main():
    syn_images_dir = '/data/mvshape/shapenetcore/single_rgb_128/'
    shapenetcore_dir = '/data/shapenetcore/ShapeNetCore.v1/'

    log.info('Getting all filenames.')
    syndirs = sorted(glob.glob(path.join(syn_images_dir, '*')))
    filenames = []
    for syndir in syndirs:
        modeldirs = sorted(glob.glob(path.join(syndir, '*')))
        if is_subset:
            modeldirs = modeldirs[:10]
        for modeldir in modeldirs:
            renderings = sorted(glob.glob(path.join(modeldir, '*.png')))
            if is_subset:
                renderings = renderings[:7]
            filenames.extend(renderings)

    # random.seed(42)
    # if not is_subset:
    #     random.shuffle(filenames)
    #     filenames = filenames[:1000000]

    random.seed(42)

    log.info('{} files'.format(len(filenames)))


    # TODO
    target_dir = '/data/mvshape/database'

    if is_subset:
        sqlite_file_path = join(target_dir, 'shapenetcore_subset.sqlite')
    else:
        sqlite_file_path = join(target_dir, 'shapenetcore.sqlite')
    output_cam_distance_from_origin = 2

    log.info('Setting up output directory.')
    # set up debugging directory.
    if path.isfile(sqlite_file_path):
        os.remove(sqlite_file_path)
    io_utils.ensure_dir_exists(target_dir)

    # used for making sure there is no duplicate.
    duplicate_name_check_set = set()

    log.info('Checking for duplicates. And making sure params.txt exists.')
    for i, filename in enumerate(filenames):
        m = re.search(r'single_rgb_128/(.*?)/(.*?)/[^_]+?_[^_]+?_v(\d{4,})_a', filename)
        synset = m.group(1)
        model_name = m.group(2)
        v = m.group(3)
        image_num = int(v)
        vc_rendering_name = '{}_{}_{:04d}'.format(synset, model_name, image_num)
        if vc_rendering_name in duplicate_name_check_set:
            print('duplicate found: ', (filename, vc_rendering_name))
        duplicate_name_check_set.add(vc_rendering_name)
        params_filename = join(syn_images_dir, synset, model_name, 'params.txt')
        assert path.isfile(params_filename)

    # Create the database
    dbm.init(sqlite_file_path)

    with dbm.db.transaction() as txn:
        log.info('Creating common objects.')
        make_dataset('shapenetcore')

        make_rendering_type('rgb')
        make_rendering_type('depth')
        make_rendering_type('normal')
        make_rendering_type('voxels')

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

        make_split('train')
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

        # Prepare all category objects.
        log.info('Preparing categories.')
        synset_db_category_map = {}
        for synset, synset_name in synset_name_pairs:
            db_category_i, _ = dbm.Category.get_or_create(name=synset_name)
            synset_db_category_map[synset] = db_category_i

        txn.commit()

        # Prepare all mesh model objects.
        # ---------------------------------------------
        db_object_map = {}
        # model_name -> {rendering_type_name -> rendering}
        db_object_centered_renderings = {}
        log.info('Preparing mesh model objects.')
        start_time = time.time()
        count = 0
        for i, filename in enumerate(filenames):
            m = re.search(r'single_rgb_128/(.*?)/(.*?)/[^_]+?_[^_]+?_v(\d{4,})_a', filename)
            synset = m.group(1)
            model_name = m.group(2)
            if model_name not in db_object_map:
                mesh_filename = join(shapenetcore_dir, synset, model_name, 'model.obj')
                assert path.isfile(mesh_filename)

                mesh_filename_suffix = join('/mesh/shapenetcore/v1', '/'.join(mesh_filename.split('/')[-3:]))

                db_category = synset_db_category_map[synset]
                # Must be unique.
                db_object = dbm.Object.create(
                    name=model_name,
                    category=db_category,
                    dataset=datasets['shapenetcore'],
                    num_vertices=0,  # Not needed for now. Easy to fill in later.
                    num_faces=0,
                    mesh_filename=mesh_filename_suffix,
                )
                db_object_map[model_name] = db_object

                oc_rendering_name = '{}_{}'.format(synset, model_name)

                assert model_name not in db_object_centered_renderings
                db_object_centered_renderings[model_name] = {
                    'output_rgb': dbm.ObjectRendering.create(
                        type=rendering_types['rgb'],
                        camera=db_oc_output_cam,
                        object=db_object,
                        # JPG
                        filename='/shapenetcore/mv20_rgb_128/{}.bin'.format(oc_rendering_name),
                        resolution=128,
                        num_channels=3,
                        set_size=20,
                        is_normalized=False,
                    ),
                    'output_depth': dbm.ObjectRendering.create(
                        type=rendering_types['depth'],
                        camera=db_oc_output_cam,
                        object=db_object,
                        # Since there is only one gt rendering per model, their id is the same as the model name.
                        filename='/shapenetcore/mv20_depth_128/{}.bin'.format(oc_rendering_name),
                        resolution=128,
                        num_channels=1,
                        set_size=20,
                        is_normalized=False,
                    ),
                    'output_normal': dbm.ObjectRendering.create(
                        type=rendering_types['normal'],
                        camera=db_oc_output_cam,
                        object=db_object,
                        filename='/shapenetcore/mv20_normal_128/{}.bin'.format(oc_rendering_name),
                        resolution=128,
                        num_channels=3,
                        set_size=20,
                        is_normalized=False,
                    ),
                    'output_voxels': dbm.ObjectRendering.create(
                        type=rendering_types['voxels'],
                        camera=db_oc_output_cam,
                        object=db_object,
                        filename='/shapenetcore/voxels_32/{}.bin'.format(oc_rendering_name),
                        resolution=32,
                        num_channels=1,
                        set_size=1,
                        is_normalized=False,
                    )
                }

                if count % 5000 == 0:
                    txn.commit()
                    t_elapsed = (time.time() - start_time)
                    t_remaining = (t_elapsed / (i + 1) * (len(filenames) - i))
                    log.info('Creating mesh objects in db. {} of {}. elapsed: {:.1f} min, remaining: {:.1f} min'.format(i, len(filenames), t_elapsed / 60, t_remaining / 60))

                count += 1
        txn.commit()
        t_elapsed = time.time() - start_time
        log.info('created {} mesh objects in db. elapsed: {:.1f} min'.format(count, t_elapsed / 60))

        start_time = time.time()

        log.info('Processing rgb images.')
        for i, filename in enumerate(filenames):
            m = re.search(r'single_rgb_128/(.*?)/(.*?)/[^_]+?_[^_]+?_v(\d{4,})_a', filename)
            synset = m.group(1)
            model_name = m.group(2)
            v = m.group(3)
            image_num = int(v)

            params_filename = join(syn_images_dir, synset, model_name, 'params.txt')
            assert path.isfile(params_filename)

            lines = render_for_cnn_utils.read_params_file(params_filename)
            Rt = render_for_cnn_utils.get_Rt_from_RenderForCNN_parameters(lines[image_num])

            # Input and output cameras
            # -------------------
            input_cam = camera.OrthographicCamera.from_Rt(Rt, wh=(128, 128), is_world_to_cam=True)
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

            vc_rendering_name = '{}_{}_{:04d}'.format(synset, model_name, image_num)

            # Viewer centered renderings.
            # --------------------------------

            # Input rgb image:
            db_object_rendering_input_rgb = dbm.ObjectRendering.create(
                type=rendering_types['rgb'],
                camera=db_input_cam,
                object=db_object,
                # This should already exist.
                filename='/shapenetcore/single_rgb_128/{}.png'.format(vc_rendering_name),
                resolution=128,
                num_channels=1,
                set_size=1,
                is_normalized=False,  # False for rgb.
            )

            db_object_rendering_input_depth = dbm.ObjectRendering.create(
                type=rendering_types['depth'],
                camera=db_input_cam_depth,
                object=db_object,
                filename='/shapenetcore/single_depth_128/{}.bin'.format(vc_rendering_name),
                resolution=128,
                num_channels=1,
                set_size=1,
                is_normalized=True,
            )

            db_object_rendering_vc_output_rgb = dbm.ObjectRendering.create(
                type=rendering_types['rgb'],
                camera=db_vc_output_cam,
                object=db_object,
                filename='/shapenetcore/mv20_rgb_128/{}.bin'.format(vc_rendering_name),
                resolution=128,
                num_channels=3,
                set_size=20,
                is_normalized=False,
            )

            db_object_rendering_vc_output_depth = dbm.ObjectRendering.create(
                type=rendering_types['depth'],
                camera=db_vc_output_cam,
                object=db_object,
                filename='/shapenetcore/mv20_depth_128/{}.bin'.format(vc_rendering_name),
                resolution=128,
                num_channels=1,
                set_size=20,
                is_normalized=False,
            )

            db_object_rendering_vc_output_normal = dbm.ObjectRendering.create(
                type=rendering_types['normal'],
                camera=db_vc_output_cam,
                object=db_object,
                filename='/shapenetcore/mv20_normal_128/{}.bin'.format(vc_rendering_name),
                resolution=128,
                num_channels=3,
                set_size=20,
                is_normalized=False,
            )

            db_object_rendering_vc_output_voxels = dbm.ObjectRendering.create(
                type=rendering_types['voxels'],
                camera=db_vc_output_cam,
                object=db_object,
                filename='/shapenetcore/voxels_32/{}.bin'.format(vc_rendering_name),
                resolution=32,
                num_channels=1,
                set_size=1,
                is_normalized=False,
            )

            # Examples
            # ----------------

            # A row in the `Example` table is just an id for many-to-many references.

            # View centered
            example_viewer_centered = dbm.Example.create()
            dbm.ExampleObjectRendering.create(example=example_viewer_centered, rendering=db_object_rendering_input_rgb)
            dbm.ExampleObjectRendering.create(example=example_viewer_centered, rendering=db_object_rendering_input_depth)
            dbm.ExampleObjectRendering.create(example=example_viewer_centered, rendering=db_object_rendering_vc_output_depth)
            dbm.ExampleObjectRendering.create(example=example_viewer_centered, rendering=db_object_rendering_vc_output_normal)
            dbm.ExampleObjectRendering.create(example=example_viewer_centered, rendering=db_object_rendering_vc_output_rgb)
            dbm.ExampleObjectRendering.create(example=example_viewer_centered, rendering=db_object_rendering_vc_output_voxels)
            dbm.ExampleDataset.create(example=example_viewer_centered, dataset=datasets['shapenetcore'])
            dbm.ExampleSplit.create(example=example_viewer_centered, split=splits['train'])
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
            dbm.ExampleDataset.create(example=example_object_centered, dataset=datasets['shapenetcore'])
            dbm.ExampleSplit.create(example=example_object_centered, split=splits['train'])
            dbm.ExampleTag.create(example=example_object_centered, tag=tags['real_world'])
            dbm.ExampleTag.create(example=example_object_centered, tag=tags['object_centered'])
            dbm.ExampleTag.create(example=example_object_centered, tag=tags['perspective_input'])
            dbm.ExampleTag.create(example=example_object_centered, tag=tags['orthographic_output'])
            dbm.ExampleTag.create(example=example_object_centered, tag=tags['novelmodel'])

            if i % 5000 == 0:
                txn.commit()
                t_elapsed = (time.time() - start_time)
                t_remaining = (t_elapsed / (i + 1) * (len(filenames) - i))
                log.info('Creating examples in db. {} of {}. elapsed: {:.1f} min, remaining: {:.1f} min'.format(i, len(filenames), t_elapsed / 60, t_remaining / 60))
        txn.commit()

    dbm.db.commit()

    t_elapsed = (time.time() - start_time)
    log.info('total elapsed: {:.1f} min'.format(t_elapsed / 60))



    # path_suffix = re.search(r'syn_images.*', filename).group(0)
    # target_file = join(target_dir, 'render_for_cnn/data/', path_suffix)
    # print(target_file)


if __name__ == '__main__':
    main()
