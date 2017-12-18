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
import mvshape.shapes
from mvshape import torch_utils
import torch
import torch.nn
import torch.autograd
import mvshape.models
import mvshape.models.encoderdecoder
from dshin import geom2d
import mvshape.models.eval_utils
import mvshape.torch_utils
import matplotlib.pyplot as pt
import mvshape.shapes
import mvshape.camera_utils
import dshin.camera
import mvshape.meshdist
import os


def main():
    base = '/data/mvshape'
    batch_size = 50
    np.random.seed(42)
    loaders_o = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/pascal3d_test_examples_opo/all_examples.cbor',
                                                    tensors_to_read=('input_rgb', 'target_depth', 'target_voxels'), shuffle=True, batch_size=batch_size)
    loaders_v = mvshape.data.dataset.ExampleLoader2('/data/mvshape/out/splits/pascal3d_test_examples_vpo/all_examples.cbor',
                                                    tensors_to_read=('input_rgb', 'target_depth', 'target_voxels'), shuffle=True, batch_size=batch_size)
    loaders = [loaders_o, loaders_v]

    both_models = [
        mvshape.models.encoderdecoder.load_model('/data/mvshape/out/pytorch/shapenetcore_rgb_mv6/opo/0/models0_00005_0018323_00109115.pth'),
        mvshape.models.encoderdecoder.load_model('/data/mvshape/out/pytorch/shapenetcore_rgb_mv6/vpo/0/models0_00005_0018323_00109115.pth'),
    ]

    for L, M, exp in zip(loaders, both_models, ['o', 'v']):
        torch_utils.recursive_module_apply(M, lambda m: m.cuda())
        torch_utils.recursive_train_setter(M, is_training=False)

        loader = L

        while True:
            next_batch = loader.next()
            if next_batch is None:
                print('END ################################')
                break
            batch_data_np = mvshape.models.encoderdecoder.prepare_data_rgb_mv(next_batch=next_batch)
            im = batch_data_np['in_image']
            helper_torch_modules = mvshape.models.encoderdecoder.build_helper_torch_modules()
            out = mvshape.models.encoderdecoder.get_final_images_from_model(M, im, helper_torch_modules=helper_torch_modules)

            masked_depth = out['masked_depth']

            base_outdir = '/data/mvshape/out/pascal3d_recon2/'

            for bi in range(len(next_batch[0])):
                image_name = path.basename(next_batch[0][bi]['input_rgb']['filename']).split('.')[0]
                recon_dir = base_outdir + '/{}/{}/'.format(exp, image_name)
                if path.isdir(recon_dir):
                    print('{} exists. skipping'.format(recon_dir))
                    continue
                eye = next_batch[0][bi]['target_camera']['eye']
                up = next_batch[0][bi]['target_camera']['up']
                lookat = next_batch[0][bi]['target_camera']['lookat']

                Rt_list = mvshape.camera_utils.make_six_views(camera_xyz=eye, object_xyz=lookat, up=up)
                cams = [dshin.camera.OrthographicCamera.from_Rt(Rt_list[i], sRt_scale=1.75, wh=(128, 128)) for i in range(len(Rt_list))]

                mv = mvshape.shapes.MVshape(masked_images=masked_depth[bi], cameras=cams)
                mv.fssr_recon(recon_dir)

                with open(recon_dir + 'info.json', 'w') as f:
                    info = next_batch[0][bi]
                    json.dump(info, f)

                gt_filename = base + next_batch[0][bi]['mesh_filename']
                print(recon_dir)
                print(gt_filename)

                gt_target = recon_dir + 'gt.off'
                if path.islink(gt_target):
                    os.remove(gt_target)
                os.symlink(gt_filename, gt_target)


if __name__ == '__main__':
    main()
