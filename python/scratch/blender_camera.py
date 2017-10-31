import shutil
import math
from pprint import pprint
import re
import bpy
import glob


## third-party code from
# https://github.com/ShapeNet/RenderForCNN/blob/master/render_pipeline/render_model_views.py

def quaternionFromYawPitchRoll(yaw, pitch, roll):
    c1 = math.cos(yaw / 2.0)
    c2 = math.cos(pitch / 2.0)
    c3 = math.cos(roll / 2.0)
    s1 = math.sin(yaw / 2.0)
    s2 = math.sin(pitch / 2.0)
    s3 = math.sin(roll / 2.0)
    q1 = c1 * c2 * c3 + s1 * s2 * s3
    q2 = c1 * c2 * s3 - s1 * s2 * c3
    q3 = c1 * s2 * c3 + s1 * c2 * s3
    q4 = s1 * c2 * c3 - c1 * s2 * s3
    return (q1, q2, q3, q4)


def camPosToQuaternion(cx, cy, cz):
    q1a = 0
    q1b = 0
    q1c = math.sqrt(2) / 2
    q1d = math.sqrt(2) / 2
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist
    t = math.sqrt(cx * cx + cy * cy)
    tx = cx / t
    ty = cy / t
    yaw = math.acos(ty)
    if tx > 0:
        yaw = 2 * math.pi - yaw
    pitch = 0
    tmp = min(max(tx * cx + ty * cy, -1), 1)
    # roll = math.acos(tx * cx + ty * cy)
    roll = math.acos(tmp)
    if cz < 0:
        roll = -roll
    # print("%f %f %f" % (yaw, pitch, roll))
    q2a, q2b, q2c, q2d = quaternionFromYawPitchRoll(yaw, pitch, roll)
    q1 = q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d
    q2 = q1b * q2a + q1a * q2b + q1d * q2c - q1c * q2d
    q3 = q1c * q2a - q1d * q2b + q1a * q2c + q1b * q2d
    q4 = q1d * q2a + q1c * q2b - q1b * q2c + q1a * q2d
    return (q1, q2, q3, q4)


def camRotQuaternion(cx, cy, cz, theta):
    theta = theta / 180.0 * math.pi
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = -cx / camDist
    cy = -cy / camDist
    cz = -cz / camDist
    q1 = math.cos(theta * 0.5)
    q2 = -cx * math.sin(theta * 0.5)
    q3 = -cy * math.sin(theta * 0.5)
    q4 = -cz * math.sin(theta * 0.5)
    return (q1, q2, q3, q4)


def quaternionProduct(qx, qy):
    a = qx[0]
    b = qx[1]
    c = qx[2]
    d = qx[3]
    e = qy[0]
    f = qy[1]
    g = qy[2]
    h = qy[3]
    q1 = a * e - b * f - c * g - d * h
    q2 = a * f + b * e + c * h - d * g
    q3 = a * g - b * h + c * e + d * f
    q4 = a * h + b * g - c * f + d * e
    return (q1, q2, q3, q4)


def obj_centened_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return (x, y, z)


## end of third party code


bpy.ops.wm.revert_mainfile()
meshfile = '/data/shapenetcore/ShapeNetCore.v1/02691156/1a04e3eab45ca15dd86060f189eb133/model.obj'
bpy.ops.import_scene.obj(filepath=meshfile)





camObj = bpy.data.objects['Camera']



db_filename = '/data/mvshape/database/shapenet.sqlite'
# shutil.copy(db_filename)

# glob.glob('/data/render_for_cnn/data/syn_images_cropped_bkg_overlaid/02691156/10155655850468db78d106ce0a280f87/')

params_file = '/data/render_for_cnn/data/syn_images/02691156/1a04e3eab45ca15dd86060f189eb133/params.txt'
with open(params_file, 'r') as f:
    content = f.read()
lines = [[float(item) for item in re.split(r'\s+', line.strip())]
         for line in re.split(r'\r?\n', content) if line]

cameras = []

azimuth_deg, elevation_deg, theta_deg, rho = lines[2]

azimuth_deg = 0
elevation_deg = 0
theta_deg = 10

# print([azimuth_deg, elevation_deg, theta_deg, rho])

cx, cy, cz = obj_centened_camera_pos(rho, azimuth_deg, elevation_deg)
q1 = camPosToQuaternion(cx, cy, cz)
q2 = camRotQuaternion(cx, cy, cz, theta_deg)
q = quaternionProduct(q2, q1)



camObj.location[0] = cx
camObj.location[1] = cy
camObj.location[2] = cz
camObj.rotation_mode = 'QUATERNION'
camObj.rotation_quaternion[0] = q[0]
camObj.rotation_quaternion[1] = q[1]
camObj.rotation_quaternion[2] = q[2]
camObj.rotation_quaternion[3] = q[3]


# M = transforms.quaternion_matrix(np.array(q))
# M[:3, 3] = [cx, cy, cz]
# Rt = M[:3]
#
# cam = camera.OrthographicCamera.from_Rt(Rt=Rt, wh=(128, 128), is_world_to_cam=True)
# cameras.append(cam)
