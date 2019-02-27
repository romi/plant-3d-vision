import numpy as np
import json

from romiscan.thirdparty import read_model

def colmap_cameras_to_json(cameras):
    res = {}
    for key in cameras.keys():
        cam = cameras[key]
        res[key] = {
            'id': cam.id,
            'model': cam.model,
            'width': cam.width,
            'height': cam.height,
            'params': cam.params.tolist()
        }
    return json.dumps(res)


def colmap_points_to_json(points):
    res = {}
    for key in points.keys():
        pt = points[key]
        res[key] = {
            'id': pt.id,
            'xyz': pt.xyz.tolist(),
            'rgb': pt.rgb.tolist(),
            'error': pt.error.tolist(),
            'image_ids': pt.image_ids.tolist(),
            'point2D_idxs': pt.point2D_idxs.tolist()
        }
    return json.dumps(res)


def colmap_points_to_pcd(points):
    n_points = len(points.keys())
    points_array = np.zeros((n_points, 3))
    colors_array = np.zeros((n_points, 3))
    for i, key in enumerate(points.keys()):
        points_array[i, :] = points[key].xyz
        colors_array[i, :] = points[key].rgb
    pass
    pcd = PointCloud()
    pcd.points = open3d.Vector3dVector(points_array)
    pcd.colors = open3d.Vector3dVector(colors_array / 255.0)
    return pcd


def colmap_images_to_json(images):
    res = {}
    for key in images.keys():
        im = images[key]
        res[key] = {
            'id': im.id,
            'qvec': im.qvec.tolist(),
            'tvec': im.tvec.tolist(),
            'rotmat': im.qvec2rotmat().tolist(),
            'camera_id': im.camera_id,
            'name': im.name,
            'xys': im.xys.tolist(),
            'point3D_ids': im.point3D_ids.tolist()
        }
    return json.dumps(res)


def cameras_model_to_opencv(cameras):
    for k in cameras.keys():
        cam = cameras[k]
        if cam['model'] == 'SIMPLE_RADIAL':
            cam['model'] = 'OPENCV'
            cam['params'] = [cam['params'][0],
                             cam['params'][0],
                             cam['params'][1],
                             cam['params'][2],
                             cam['params'][3],
                             cam['params'][3],
                             0.,
                             0.]
        elif cam['model'] == 'RADIAL':
            cam['model'] = 'OPENCV'
            cam['params'] = [cam['params'][0],
                             cam['params'][0],
                             cam['params'][1],
                             cam['params'][2],
                             cam['params'][3],
                             cam['params'][4],
                             0.,
                             0.]
        elif cam['model'] == 'OPENCV':
            pass
        else:
            raise Exception('Cannot convert cam model to opencv')
        break
    return cameras
