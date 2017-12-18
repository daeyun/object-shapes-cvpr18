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
from dshin import geom3d
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
from dshin import geom2d
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

    exps = ['o', 'v']


    # #### TODO
    # mode = 1
    #
    # loaders = [loaders[mode]]
    # both_models = [both_models[mode]]
    # exps = [exps[mode]]
    # ####


    counter = 0

    for L, M, exp in zip(loaders, both_models, exps):
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

            recon_basedir = '/data/mvshape/out/pascal3d_recon/'
            out_basedir = '/data/mvshape/out/pascal3d_figures/'

            for bi in range(len(next_batch[0])):
                image_name = path.basename(next_batch[0][bi]['input_rgb']['filename']).split('.')[0]
                recon_dir = recon_basedir + '/{}/{}/'.format(exp, image_name)
                # if path.isdir(recon_dir):
                #     print('{} exists. skipping'.format(recon_dir))
                #     continue
                eye = next_batch[0][bi]['target_camera']['eye']
                up = next_batch[0][bi]['target_camera']['up']
                lookat = next_batch[0][bi]['target_camera']['lookat']

                Rt_list = mvshape.camera_utils.make_six_views(camera_xyz=eye, object_xyz=lookat, up=up)
                cams = [dshin.camera.OrthographicCamera.from_Rt(Rt_list[i], sRt_scale=1.75, wh=(128, 128)) for i in range(len(Rt_list))]

                mv = mvshape.shapes.MVshape(masked_images=masked_depth[bi], cameras=cams)

                depth_mesh_filenames = glob.glob(recon_dir + 'depth_meshes/*.ply')
                pcl = []
                for item in depth_mesh_filenames:
                    pcl.append(io_utils.read_mesh(item)['v'])
                pcl = np.concatenate(pcl, axis=0)
                print(pcl.shape)

                fig_dir = path.join(out_basedir, '{}/{}/'.format(exp, image_name))
                io_utils.ensure_dir_exists(fig_dir)


                pt.figure(figsize=(5, 5))
                ax = pt.gca(projection='3d')
                color = pcl[:, 0] + 0.6  # +0.6 to force the values to be positive. not necessary.
                rotmat = transforms.rotation_matrix(angle=45, direction=np.array((0, 1, 0)))
                rotmat2 = transforms.rotation_matrix(angle=-30, direction=np.array((0, 0, 1)))
                pcl = transforms.apply44(rotmat2.dot(rotmat), pcl)
                index_array = np.argsort(pcl[:,0])
                pcl = pcl[index_array]
                color = color[index_array]
                geom3d.pts(pcl, markersize=45, color=color, zdir='y', show_labels=False, cmap='viridis', cam_sph=(1, 90, 0), ax=ax)
                ax.axis('off')
                pt.savefig(fig_dir+'pcl.png', bbox_inches='tight', transparent=True, pad_inches=0)
                pt.close()


                pt.figure(figsize=(5, 5))
                ax = pt.gca()
                geom2d.draw_depth(out['silhouette_prob'][bi], cmap='gray', nan_color=(1.0, 1.0, 1.0), grid=128, grid_width=3, ax=ax, show_colorbar=False, show_colorbar_ticks=False)
                pt.savefig(fig_dir+'/silhouette.png', bbox_inches='tight', transparent=False, pad_inches=0)
                pt.close()

                pt.figure(figsize=(10, 10))
                ax = pt.gca()
                geom2d.draw_depth(masked_depth[bi], cmap='viridis', nan_color=(1.0, 1.0, 1.0), grid=128, grid_width=6, ax=ax, show_colorbar=False, show_colorbar_ticks=False)
                pt.savefig(fig_dir+'/masked-depth.png', bbox_inches='tight', transparent=False, pad_inches=0)
                pt.close()

                rgb_filename = base+next_batch[0][bi]['input_rgb']['filename']
                assert path.isfile(rgb_filename)

                rgb_link_target = fig_dir + '/input.png'
                if path.islink(rgb_link_target):
                    os.remove(rgb_link_target)
                os.symlink(rgb_filename, rgb_link_target)
                print(counter, fig_dir, rgb_filename)

                counter += 1




if __name__ == '__main__':
    main()
