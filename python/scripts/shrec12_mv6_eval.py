import numpy as np
from os import path
import sys
from dshin import transforms
import glob
from mvshape import pascal
from mvshape import io_utils
from dshin import vtk_utils
import collections
import json
import mvshape.data
import mvshape.data.dataset
import mvshape.data.parallel_reader
import mvshape.models.encoderdecoder
from mvshape import torch_utils
import torch
import torch.nn
import torch.autograd
import mvshape.models
import mvshape.shapes
import mvshape.models.encoderdecoder
import mvshape.models.eval_utils
import mvshape.torch_utils
import mvshape.camera_utils
import dshin.camera
import mvshape.meshdist
import os


def main2():
    batch_size = 50
    loaders = {}
    loaders['novelview_o'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelview_examples_opo/all_examples.cbor',
                                                                 tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')
    loaders['novelview_v'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelview_examples_vpo/all_examples.cbor',
                                                                 tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')
    loaders['novelmodel_o'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelmodel_examples_opo/all_examples.cbor',
                                                                  tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')
    loaders['novelmodel_v'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelmodel_examples_vpo/all_examples.cbor',
                                                                  tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')
    loaders['novelclass_o'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelclass_examples_opo/all_examples.cbor',
                                                                  tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')
    loaders['novelclass_v'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelclass_examples_vpo/all_examples.cbor',
                                                                  tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')

    next_batch_list = collections.defaultdict(list)
    for exp in ['novelview', 'novelmodel', 'novelclass']:
        for exp2 in ['o', 'v']:
            name = '{}_{}'.format(exp, exp2)
            while True:
                next_batch = loaders[name].next()
                if next_batch is None:
                    break
                next_batch_list[name].append(next_batch)
            assert len(next_batch_list[name]) == 600 // batch_size

    results_to_save = collections.defaultdict(lambda: collections.defaultdict(list))
    epochs = [25]
    for epoch in epochs:
        for exp in ['novelview', 'novelmodel', 'novelclass']:
            for exp2 in ['vpo', 'opo']:
                files = glob.glob('/data/mvshape/out/pytorch/shrec12_voxels/{}/0/models0_{:05d}_*'.format(exp2, epoch))
                assert len(files) == 1
                models = mvshape.models.encoderdecoder.load_model(files[0])
                torch_utils.recursive_module_apply(models, lambda m: m.cuda())
                torch_utils.recursive_train_setter(models, is_training=False)
                helper_torch_modules = mvshape.models.encoderdecoder.build_helper_torch_modules()

                name = '{}_{}'.format(exp, exp2[0])
                out_list = []
                for next_batch in next_batch_list[name]:
                    batch_data_np = mvshape.models.encoderdecoder.prepare_data_depth_voxels(next_batch=next_batch)
                    batch_data = torch_utils.recursive_numpy_to_torch(batch_data_np, cuda=True)
                    out = mvshape.models.encoderdecoder.get_output_from_model(models, batch_data, helper_torch_modules, include_output_tensors=True)
                    out_list.append(torch_utils.recursive_torch_to_numpy(out))
                    print('.', end='')
                print(name)
                results_to_save[epoch][name].append(out_list)

    np.save('/data/mvshape/out/shrec12_voxel_results.npy', dict(results_to_save))


def main3():
    batch_size = 600
    loaders = {}
    loaders['novelview_o'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelview_examples_opo/all_examples.cbor',
                                                                 tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')
    loaders['novelview_v'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelview_examples_vpo/all_examples.cbor',
                                                                 tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')
    loaders['novelmodel_o'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelmodel_examples_opo/all_examples.cbor',
                                                                  tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')
    loaders['novelmodel_v'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelmodel_examples_vpo/all_examples.cbor',
                                                                  tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')
    loaders['novelclass_o'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelclass_examples_opo/all_examples.cbor',
                                                                  tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')
    loaders['novelclass_v'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelclass_examples_vpo/all_examples.cbor',
                                                                  tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')

    next_batch = collections.defaultdict(list)
    for exp in ['novelview', 'novelmodel', 'novelclass']:
        for exp2 in ['o', 'v']:
            name = '{}_{}'.format(exp, exp2)
            next_batch[name] = loaders[name].next()
            assert len(next_batch[name][0]) == 600, len(next_batch[name])

    results = np.load('/data/mvshape/out/shrec12_voxel_results_25.npy')

    for exp in ['novelview', 'novelmodel', 'novelclass']:
        for exp2 in ['vpo', 'opo']:
            name = '{}_{}'.format(exp, exp2[0])
            # print(next_batch[name][1]['target_voxels'].shape)
            # t = next_batch[name][1]['target_voxels'] > 0.5
            # o = results[()][name] > 0
            # (t&o).sum(axis=2, keep)
            # print(results[()][name].shape)


def main4():
    batch_size = 50
    loaders = {}
    loaders['novelview_o'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelview_examples_opo/all_examples.cbor',
                                                                 tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')
    loaders['novelview_v'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelview_examples_vpo/all_examples.cbor',
                                                                 tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')
    loaders['novelmodel_o'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelmodel_examples_opo/all_examples.cbor',
                                                                  tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')
    loaders['novelmodel_v'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelmodel_examples_vpo/all_examples.cbor',
                                                                  tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')
    loaders['novelclass_o'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelclass_examples_opo/all_examples.cbor',
                                                                  tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')
    loaders['novelclass_v'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelclass_examples_vpo/all_examples.cbor',
                                                                  tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')

    next_batch_list = collections.defaultdict(list)
    for exp in ['novelview', 'novelmodel', 'novelclass']:
        for exp2 in ['o', 'v']:
            name = '{}_{}'.format(exp, exp2)
            while True:
                next_batch = loaders[name].next()
                if next_batch is None:
                    break
                next_batch_list[name].append(next_batch)
            assert len(next_batch_list[name]) == 600 // batch_size

    helper_torch_modules = mvshape.models.encoderdecoder.build_helper_torch_modules()
    results_to_save = collections.defaultdict(lambda: collections.defaultdict(list))
    epochs = [45]
    for epoch in epochs:
        for exp in ['novelview', 'novelmodel', 'novelclass']:
            for exp2 in ['vpo', 'opo']:
                files = glob.glob('/data/mvshape/out/pytorch/shrec12_depth_mv6/{}/1/models0_{:05d}_*'.format(exp2, epoch))
                assert len(files) == 1
                models = mvshape.models.encoderdecoder.load_model(files[0])
                torch_utils.recursive_module_apply(models, lambda m: m.cuda())
                torch_utils.recursive_train_setter(models, is_training=False)

                name = '{}_{}'.format(exp, exp2[0])
                out_list = []
                print(len(next_batch_list[name]))
                for next_batch in next_batch_list[name]:
                    batch_data_np = mvshape.models.encoderdecoder.prepare_data_depth_mv(next_batch=next_batch)
                    batch_data = torch_utils.recursive_numpy_to_torch(batch_data_np, cuda=True)
                    out = mvshape.models.encoderdecoder.get_output_from_model(models, batch_data, helper_torch_modules, include_output_tensors=True)
                    out = torch_utils.recursive_torch_to_numpy(out)
                    out_list.append((out, next_batch))
                results_to_save[epoch][name].append(out_list)

    np.save('/data/mvshape/out/shrec12_mv_results.npy', dict(results_to_save))

def main():
    batch_size = 50
    loaders = {}
    loaders['novelview_o'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelview_examples_opo/all_examples.cbor',
                                                                 tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')
    loaders['novelview_v'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelview_examples_vpo/all_examples.cbor',
                                                                 tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')
    loaders['novelmodel_o'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelmodel_examples_opo/all_examples.cbor',
                                                                  tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')
    loaders['novelmodel_v'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelmodel_examples_vpo/all_examples.cbor',
                                                                  tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')
    loaders['novelclass_o'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelclass_examples_opo/all_examples.cbor',
                                                                  tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')
    loaders['novelclass_v'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelclass_examples_vpo/all_examples.cbor',
                                                                  tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')

    next_batch_list = collections.defaultdict(list)
    for exp in ['novelview', 'novelmodel', 'novelclass']:
        for exp2 in ['o', 'v']:
            name = '{}_{}'.format(exp, exp2)
            while True:
                next_batch = loaders[name].next()
                if next_batch is None:
                    break
                next_batch_list[name].append(next_batch)
            assert len(next_batch_list[name]) == 600 // batch_size

    helper_torch_modules = mvshape.models.encoderdecoder.build_helper_torch_modules()
    results_to_save = collections.defaultdict(lambda: collections.defaultdict(list))
    epochs = [45]
    for epoch in epochs:
        for exp in ['novelview', 'novelmodel', 'novelclass']:
            for exp2 in ['vpo', 'opo']:
                files = glob.glob('/data/mvshape/out/pytorch/shrec12_depth_mv6/{}/1/models0_{:05d}_*'.format(exp2, epoch))
                assert len(files) == 1
                models = mvshape.models.encoderdecoder.load_model(files[0])
                torch_utils.recursive_module_apply(models, lambda m: m.cuda())
                torch_utils.recursive_train_setter(models, is_training=False)

                name = '{}_{}'.format(exp, exp2[0])
                out_list = []
                out_list2 = []
                print(len(next_batch_list[name]))
                for next_batch in next_batch_list[name]:
                    batch_data_np = mvshape.models.encoderdecoder.prepare_data_depth_mv(next_batch=next_batch)
                    batch_data = torch_utils.recursive_numpy_to_torch(batch_data_np, cuda=True)
                    out = mvshape.models.encoderdecoder.get_output_from_model(models, batch_data, helper_torch_modules, include_output_tensors=True)
                    out = torch_utils.recursive_torch_to_numpy(out)
                    out_list.append(out['loss']['silhouette'])
                    out_list2.append(out['loss']['depth'])
                print(exp, exp2, np.mean(out_list), np.mean(out_list2))

    np.save('/data/mvshape/out/shrec12_mv_results.npy', dict(results_to_save))


def main5():
    base = '/data/mvshape'
    batch_size = 30
    loaders = {}
    loaders['novelview_o'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelview_examples_opo/all_examples.cbor',
                                                                 tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')
    loaders['novelview_v'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelview_examples_vpo/all_examples.cbor',
                                                                 tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')
    loaders['novelmodel_o'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelmodel_examples_opo/all_examples.cbor',
                                                                  tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')
    loaders['novelmodel_v'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelmodel_examples_vpo/all_examples.cbor',
                                                                  tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')
    loaders['novelclass_o'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelclass_examples_opo/all_examples.cbor',
                                                                  tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')
    loaders['novelclass_v'] = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/shrec12_novelclass_examples_vpo/all_examples.cbor',
                                                                  tensors_to_read=('input_depth', 'target_depth', 'target_voxels'), shuffle=False, batch_size=batch_size, split_name='TEST')

    next_batch_list = collections.defaultdict(list)
    for exp in ['novelview', 'novelmodel', 'novelclass']:
        for exp2 in ['o', 'v']:
            name = '{}_{}'.format(exp, exp2)
            while True:
                next_batch = loaders[name].next()
                if next_batch is None:
                    break
                next_batch_list[name].append(next_batch)
            assert len(next_batch_list[name]) == 600 // batch_size

    helper_torch_modules = mvshape.models.encoderdecoder.build_helper_torch_modules()
    results_to_save = collections.defaultdict(lambda: collections.defaultdict(list))
    epochs = [45]
    base_outdir = '/data/mvshape/out/shrec12_mv6_pcl/'
    for epoch in epochs:
        for exp in ['novelview', 'novelmodel', 'novelclass']:
            for exp2 in ['vpo', 'opo']:
                files = glob.glob('/data/mvshape/out/pytorch/shrec12_depth_mv6/{}/1/models0_{:05d}_*'.format(exp2, epoch))
                assert len(files) == 1
                models = mvshape.models.encoderdecoder.load_model(files[0])
                torch_utils.recursive_module_apply(models, lambda m: m.cuda())
                torch_utils.recursive_train_setter(models, is_training=False)

                counter = 0

                name = '{}_{}'.format(exp, exp2[0])
                print(len(next_batch_list[name]))
                for batch_i, next_batch in enumerate(next_batch_list[name]):
                    batch_data_np = mvshape.models.encoderdecoder.prepare_data_depth_mv(next_batch=next_batch)
                    im = batch_data_np['in_image']
                    out = mvshape.models.encoderdecoder.get_final_images_from_model(models, im, helper_torch_modules)

                    for bi in range(len(next_batch[0])):
                        pcl_ourdir = base_outdir + '{}/{}/{:04d}/'.format(exp2, exp, counter)
                        if path.isdir(pcl_ourdir):
                            print(pcl_ourdir, 'exists skipping.')
                            counter += 1
                            continue
                        io_utils.ensure_dir_exists(pcl_ourdir)

                        eye = next_batch[0][bi]['target_camera']['eye']
                        up = next_batch[0][bi]['target_camera']['up']
                        lookat = next_batch[0][bi]['target_camera']['lookat']

                        Rt_list = mvshape.camera_utils.make_six_views(camera_xyz=eye, object_xyz=lookat, up=up)
                        # the 0.4 is for shrec12
                        cams = [dshin.camera.OrthographicCamera.from_Rt(Rt_list[i], sRt_scale=0.4, wh=(128, 128)) for i in range(len(Rt_list))]

                        pts_list = []
                        normals_list = []

                        for i in range(6):
                            d = out['masked_depth'][bi][i].copy()
                            mask = ~np.isnan(d)
                            d[~mask] = 0

                            pts, normals = transforms.oriented_pcl_from_ortho_depth(d, mask)
                            pts = cams[i].cam_to_world(pts)
                            normals = cams[i].cam_to_world(normals)

                            pts_list.append(pts)
                            normals_list.append(normals)

                        vpo_pts = np.concatenate(pts_list, axis=0)
                        vpo_normals = np.concatenate(normals_list, axis=0)

                        io_utils.save_points_ply(pcl_ourdir + 'pts.ply', points=vpo_pts, normals=vpo_normals, values=np.ones(vpo_pts.shape[0]), confidence=np.ones(vpo_pts.shape[0]), scale_normals_by_confidence=False, text=True)

                        gt_filename = base + next_batch[0][bi]['mesh_filename']
                        assert path.isfile(gt_filename)
                        print(gt_filename)
                        if path.islink(pcl_ourdir + 'gt.off'):
                            os.remove(pcl_ourdir + 'gt.off')
                        os.symlink(gt_filename, pcl_ourdir + 'gt.off')

                        with open(pcl_ourdir + 'info.json', 'w') as f:
                            json.dump(next_batch[0][bi], f)

                        print('.', end='', flush=True)
                        counter += 1


def main9():
    base_outdir = '/data/mvshape/out/shrec12_mv6_pcl/'
    for exp in ['novelview', 'novelmodel', 'novelclass']:
        for exp2 in ['vpo', 'opo']:
            distances = []
            distances2 = []
            for i in range(600):
                pcl_ourdir = base_outdir + '{}/{}/{:04d}/'.format(exp2, exp, i)
                assert path.isdir(pcl_ourdir)
                gt_file = pcl_ourdir + 'gt.off'
                pcl_file = pcl_ourdir + 'pts.ply'
                # mesh_to_pcl not used.
                pcl_to_mesh, mesh_to_pcl = mvshape.meshdist.between_mesh_and_pcl(gt_file, pcl_file)
                distances.append(pcl_to_mesh)
                distances2.append(mesh_to_pcl)
                print(i, end=' ', flush=True)
            print('##################################################################')
            print(exp, exp2, np.mean(distances), np.mean(distances2))


if __name__ == '__main__':
    main()
