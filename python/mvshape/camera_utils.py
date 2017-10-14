from dshin import camera
from dshin import transforms

import numpy as np
import numpy.linalg as la


def make_six_views(camera_xyz, object_xyz, up):
    camera_xyz = np.array(camera_xyz).ravel()
    object_xyz = np.array(object_xyz).ravel()
    up = np.array(up).ravel()

    viewing_dir = object_xyz - camera_xyz
    viewing_dir /= la.norm(viewing_dir)
    left = np.cross(up, viewing_dir)

    # front
    Rt_list = [
        transforms.lookat_matrix(
            cam_xyz=camera_xyz,
            obj_xyz=object_xyz,
            up=up
        )
    ]

    def rotate(angle, axis):
        rot = transforms.rotation_matrix(angle, axis, deg=True)
        new_cam_xyz = transforms.apply44(rot, camera_xyz[None, :]).ravel()
        new_up = transforms.apply44(rot, up[None, :]).ravel()
        return transforms.lookat_matrix(
            cam_xyz=new_cam_xyz,
            obj_xyz=object_xyz,
            up=new_up
        )

    Rt_list.append(rotate(180, up))  # back
    Rt_list.append(rotate(-90, up))  # left
    Rt_list.append(rotate(90, up))  # right
    Rt_list.append(rotate(90, left))  # top

    # bottom
    rot = transforms.rotation_matrix(90, left, deg=True)
    rot2 = transforms.rotation_matrix(180, viewing_dir, deg=True)
    new_cam_xyz = transforms.apply44(rot2.dot(rot), camera_xyz[None, :]).ravel()
    new_up = transforms.apply44(rot, up[None, :]).ravel()
    Rt_list.append(transforms.lookat_matrix(
        cam_xyz=new_cam_xyz,
        obj_xyz=object_xyz,
        up=new_up
    ))

    return Rt_list
