import copy
import plyfile
import typing
import glob
import shutil
from os import path

import numpy as np

import matplotlib.pyplot as pt
import itertools
import torch
import torchvision
import collections

import mvshape
from mvshape import io_utils
from mvshape import log
from mvshape import camera_utils
from mvshape.proto import dataset_pb2
from dshin import geom2d
from dshin import transforms
from dshin import geom3d
from dshin import camera
from mvshape import mve
from mvshape import pcl_utils
from mvshape import mesh_utils


class MVshape(object):
    def __init__(self, masked_images: np.ndarray, cameras: typing.Sequence[camera.OrthographicCamera]):
        assert masked_images.ndim == 3
        assert masked_images.shape[0] in (6, 20)
        assert masked_images.dtype == np.float32

        # The top-left-most pixel value is nan.
        # There might be exceptions to this, but then the dataset is likely wrong.
        assert np.isnan(masked_images.flat[0])
        self.masked_images = masked_images
        self.cameras = cameras

        # Assume all cameras have the same zoom factor.
        for cam in cameras[1:]:
            assert cam.sRt_scale == cameras[0].sRt_scale

    def plot_images(self):
        geom2d.draw_depth(self.masked_images, in_order='chw', grid=self.masked_images.shape[-1], grid_width=min(self.masked_images.shape[0], 10))

    def _depth_meshes(self, out_dir):
        mve.save_as_mve_views(out_dir, masked_depth_images=self.masked_images, ortho_cameras=self.cameras, cam_scale=self.cameras[0].sRt_scale)
        meshdir = mve.convert_mve_views_to_meshes(out_dir)
        ply_files = sorted(glob.glob(path.join(meshdir, 'mesh_*.ply')))
        assert path.split(ply_files[0])[-1] == 'mesh_0000.ply', ply_files

        # if self._use_input_view:
        #     fname = ply_files[0]
        #     plydata = plyfile.PlyData.read(fname)
        #
        #     pts = np.vstack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'])).T
        #
        #     cam0 = self.cameras[0]
        #     cam0_pts = cam0.world_to_cam(pts)
        #     cam0_pts[:, 2] -= self._input_view_z_visibility_offset
        #
        #     cam0_pts_scaled = (cam0_pts - self._input_view_normalized_centroid_xyz) * self._input_view_scale + self._input_view_normalized_centroid_xyz
        #     cam0_pts_transformed = cam0_pts_scaled + self._input_view_offset_xyz
        #     pts_transformed = cam0.cam_to_world(cam0_pts_transformed)
        #
        #     plydata['vertex']['x'] = pts_transformed[:, 0]
        #     plydata['vertex']['y'] = pts_transformed[:, 1]
        #     plydata['vertex']['z'] = pts_transformed[:, 2]
        #
        #     shutil.copyfile(fname, path.join(path.dirname(fname), '{}.bak'.format(path.split(fname)[-1])))
        #
        #     plydata.write(fname)

        return ply_files

    def depth_meshes(self, out_dir):
        ply_files = self._depth_meshes(out_dir)
        assert len(ply_files) == self.masked_images.shape[0]

        meshes = []
        for i, file in enumerate(ply_files):
            meshes.append(MeshShape(file, cam=camera.OrthographicCamera.identity(wh=self.masked_images.shape[1:][::-1])))

        return meshes

    def fssr_recon(self, out_dir):
        ply_files = self._depth_meshes(out_dir=out_dir)

        fssr_recon_file = mve.fssr_pcl_files(ply_files, scale=0.3)
        fssr_recon_clean_file = mve.meshclean(fssr_recon_file, threshold=0.25)

        recon_dir = io_utils.ensure_dir_exists(path.join(out_dir, 'recon'))

        new_fssr_recon_file = path.join(recon_dir, path.basename(fssr_recon_file))
        new_fssr_recon_clean_file = path.join(recon_dir, path.basename(fssr_recon_clean_file))

        shutil.move(fssr_recon_file, new_fssr_recon_file)
        shutil.move(fssr_recon_clean_file, new_fssr_recon_clean_file)

    def fssr_recon_using_input(self, out_dir, aligned_depth_mesh_filename):
        ply_files = [aligned_depth_mesh_filename] + sorted(glob.glob(path.join(out_dir, 'depth_meshes/*')))[1:]

        fssr_recon_file = mve.fssr_pcl_files(ply_files, scale=0.3)
        fssr_recon_clean_file = mve.meshclean(fssr_recon_file, threshold=0.25)

        recon_dir = io_utils.ensure_dir_exists(path.join(out_dir, 'recon'))

        new_fssr_recon_file = path.join(recon_dir, path.basename(fssr_recon_file))
        new_fssr_recon_clean_file = path.join(recon_dir, path.basename(fssr_recon_clean_file))

        shutil.move(fssr_recon_file, new_fssr_recon_file + '.fused.ply')
        shutil.move(fssr_recon_clean_file, new_fssr_recon_clean_file + '.fused.ply')

    def save_depth_mesh_vertices(self, out_dir):
        depth_meshes = self.depth_meshes(out_dir)
        all_vertices = np.concatenate([item.fv['v'] for item in depth_meshes], axis=0)
        vertices_file = path.join(out_dir, 'pcl/vertices.ply')
        io_utils.save_simple_points_ply(vertices_file, all_vertices)


class MeshShape(object):
    def __init__(self, mesh_or_path, cam: camera.Camera = None):
        if isinstance(mesh_or_path, str):
            try:
                mesh_or_path = io_utils.read_mesh(mesh_or_path)
                if mesh_or_path['v'].size == 0:
                    raise ValueError('Empty mesh.')
            except Exception as ex:
                log.error('Error reading mesh {}'.format(mesh_or_path))
                raise ex
        assert isinstance(mesh_or_path, dict)
        self.fv_world = copy.deepcopy(mesh_or_path)
        self.fv = self.fv_world
        self._is_matlab_path_initialized = False

        if cam is None:
            identity_Rt = np.eye(3, 4, dtype=np.float64)
            cam = camera.OrthographicCamera.from_Rt(identity_Rt)
        self.cam = cam

    def apply_34(self, M):
        assert M.shape == (3, 4)
        v_cam = transforms.apply34(M, self.fv_world['v'])
        self.fv['v'] = v_cam

    def plot(self,
             cams: typing.Union[camera.Camera, typing.Sequence[camera.Camera]] = None,
             ax=None):
        if cams is not None:
            if not isinstance(cams, (list, tuple)):
                cams = [cams]
            ax = geom3d.draw_cameras(cams, ax=ax)
        ax = geom3d.plot_mesh(self.fv, ax=ax)
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        if cams is not None:
            cam = cams[0]
            assert isinstance(cam, camera.Camera)
            sph = transforms.xyz_to_sph(cam.pos).ravel()
            incl = np.rad2deg(sph[1])
            az = np.rad2deg(sph[2])
            ax.view_init(elev=90 - float(incl), azim=float(az))
        return ax


class MVShapeResultsShrec12(object):
    """
    Wrapper around all inputs, outputs, targets, and metadata.
    """

    def __init__(self, all_out_dict: typing.Dict):
        assert isinstance(all_out_dict, dict)
        self._results = all_out_dict

    def visualize_output(self, tag, i):
        s = self.new_mvshape_object_from_target(tag, i)
        s.plot_images()

    def visualize_target(self, tag: str, i: int):
        s = self.new_mvshape_object_from_output(tag, i)
        s.plot_images()

    def visualize_input(self, tag: str, i: int):
        # (1, 128, 128), masked
        images = self._results[tag]['in_images'][i]
        assert images.shape[0] == 1
        geom2d.draw_depth(images, in_order='hw', grid=images.shape[-1], grid_width=min(images.shape[0], 10))

    def target_camera_objects(self, tag: str, i: int):
        example = self._results[tag]['examples'][i]
        eye = np.array(example.multiview_depth.eye)
        up = np.array(example.multiview_depth.up)
        lookat = np.array(example.multiview_depth.lookat)

        Rt_list = camera_utils.make_six_views(eye, lookat, up)

        # Important: scale factor in camera coordinates, after Rt is applied.
        scale = example.multiview_depth.scale
        assert isinstance(scale, float)

        resolution = example.multiview_depth.resolution

        cameras = []
        for Rt in Rt_list:
            cam = camera.OrthographicCamera.from_Rt(Rt=Rt, wh=(resolution, resolution),
                                                    sRt_scale=scale, is_world_to_cam=True)
            cameras.append(cam)

        return cameras

    def new_mvshape_object_from_output(self, tag: str, i: int):
        # (6, 128, 128)
        images = self._results[tag]['out_images'][i]
        assert images.shape[0] == 6
        return MVshape(images, cameras=self.target_camera_objects(tag, i))

    def new_mvshape_object_from_target(self, tag: str, i: int):
        # (6, 128, 128)
        images = self._results[tag]['target_images'][i]
        assert images.shape[0] == 6
        return MVshape(images, cameras=self.target_camera_objects(tag, i))

    def mesh_filename(self, tag: str, i: int):
        example = self._results[tag]['examples'][i]
        filename = example.multiview_depth.mesh_filename
        mesh_filename = mvshape.make_data_path(filename)
        return mesh_filename

    def input_image(self, tag, i):
        images = self._results[tag]['in_images'][i]
        assert images.shape[0] == 1
        return images.squeeze()

    def save_input_image_as_mesh(self, tag, i, out_dir):
        images = self._results[tag]['in_images'][i]
        assert images.shape[0] == 1
        # target view-0. same viewpoint, but different (x,y,z) offset and scale in camera space.
        camera = self.target_camera_objects(tag, i)[0]

        scaled_images = images * camera.sRt_scale

        # negative valued vertices are cropped. So make them positive.
        minimum = scaled_images[~np.isnan(scaled_images)].min()
        scaled_images -= minimum
        # arbitrary offset to make sure camera is not too close.
        scaled_images += 1.0

        mve.save_as_mve_views(out_dir, masked_depth_images=scaled_images, ortho_cameras=[camera], cam_scale=camera.sRt_scale)
        # mve.save_as_mve_views(out_dir, masked_depth_images=scaled_images, ortho_cameras=[camera.OrthographicCamera.identity(wh=cam.wh)], cam_scale=cam.sRt_scale)
        meshdir = mve.convert_mve_views_to_meshes(out_dir)
        ply_files = sorted(glob.glob(path.join(meshdir, 'mesh_*.ply')))
        return ply_files

    def save_input_image_as_aligned_mesh(self, tag, i, outdir):
        self.save_input_image_as_mesh(tag, i, path.join(outdir, 'input'))
        saved_mesh = path.join(outdir, 'input/depth_meshes/mesh_0000.ply')
        out0_mesh = path.join(outdir, 'depth_meshes/mesh_0000.ply')
        target_mesh = path.join(outdir, 'input/transformed_input.ply')

        # view0 camera
        cam = self.target_camera_objects(tag, i)[0]

        tmp_filename0 = io_utils.temp_filename('/tmp/mvshape_tmp', suffix='_transformed.ply')
        tmp_filename1 = io_utils.temp_filename('/tmp/mvshape_tmp', suffix='_transformed.ply')

        io_utils.ensure_dir_exists(path.dirname(tmp_filename0))
        io_utils.ensure_dir_exists(path.dirname(tmp_filename1))

        Rt = cam.Rt()
        mesh_utils.transform_ply(saved_mesh, tmp_filename0, Rt)
        mesh_utils.transform_ply(out0_mesh, tmp_filename1, Rt)

        source = io_utils.read_ply_pcl(tmp_filename0)['v']
        target = io_utils.read_ply_pcl(tmp_filename1)['v']

        offset, scale = pcl_utils.find_aligning_transformation(source, target)

        M = np.eye(4)
        M[:3, :3] *= scale
        M[:3, 3] = offset

        Rt44 = np.eye(4)
        Rt44[:3, :] = Rt

        final_transform = cam.Rt_inv().dot(M.dot(Rt44))

        mesh_utils.transform_ply(saved_mesh, target_mesh, final_transform, confidence_scale=2.0, value_scale=1.7)

        return target_mesh

    def new_mesh_object(self, tag: str, i: int):
        mesh_filename = self.mesh_filename(tag, i)
        return MeshShape(mesh_or_path=mesh_filename)
