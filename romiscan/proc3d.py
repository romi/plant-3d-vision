"""
romiscan.proc3d
---------------

This module contains all functions for processing of 3D data.

"""
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import os
from scipy.ndimage.filters import gaussian_filter

from romiscan import cgal

from open3d.geometry import PointCloud, TriangleMesh
import imageio
from open3d.utility import Vector3dVector, Vector3iVector
import open3d
import cv2
import proc2d

def index2point(indexes, origin, voxel_size):
    """Converts discrete nd indexes to a 3d points

    Parameters
    __________
    indexes : np.ndarray
        Nxd array of indices
    origin : np.ndarray
        1d array of length d
    voxel_size : float
        size of voxels

    Returns
    _______
    np.ndarray
        Nxd array of points
    """
    return voxel_size * indexes + origin[np.newaxis, :]

def point2index(points, origin, voxel_size):
    """Converts discrete nd indexes to a 3d points

    Parameters
    __________
    points : np.ndarray
        Nxd array of points
    origin : np.ndarray
        1d array of length d
    voxel_size : float
        size of voxels

    Returns
    _______
    np.ndarray (dtype=int)
        Nxd array of indices
    """
    return np.array(np.round((points - origin[np.newaxis, :]) / voxel_size), dtype=int)

def pcd2mesh(pcd):
    """Use CGAL to create a Delaunay triangulation of a point cloud
    with normals.

    Parameters
    __________
    pcd: open3d.geometry.PointCloud
        input point cloud (must have normals)

    Returns
    _______
    open3d.geometry.TriangleMesh
    """
    assert(pcd.has_normals)
    points, triangles = cgal.poisson_mesh(np.asarray(pcd.points),
                          np.asarray(pcd.normals))

    mesh = TriangleMesh()
    mesh.vertices = Vector3dVector(points)
    mesh.triangles = Vector3iVector(triangles)

    return mesh


def pcd2vol(pcd, voxel_size, zero_padding=0):
    """
    Voxelize a point cloud. Every voxel value is equal to the number
    of points in the corresponding cube.

    Parameters
    __________
    pcd : open3d.geometry.PointCloud
        input point cloud
    voxel_size : float
        target voxel size
    zero_padding : int
        number of zero padded values on every side of the volume (default = 0)

    Returns
    _______
    vol : np.ndarray
    origin : list
    """
    pcd_points = np.asarray(pcd.points)
    origin = np.min(pcd_points, axis=0) - zero_padding*voxel_size
    indices = point2index(pcd_points, origin, voxel_size)
    shape = indices.max(axis=0)

    vol = np.zeros(shape + 2*zero_padding + 1, dtype=np.float)
    indices = indices + zero_padding

    for i in range(pcd_points.shape[0]):
        vol[indices[i, 0], indices[i, 1], indices[i, 2]] += 1.

    return vol, origin

def skeletonize(mesh):
    """Use CGAL to create a Delaunay triangulation of a point cloud
    with normals.

    Parameters
    __________
    mesh: open3d.geometry.TriangleMesh
        input mesh

    Returns
    _______
    json
    """
    points, lines = cgal.skeletonize_mesh(
        np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    return {'points': points.tolist(),
            'lines': lines.tolist()}


def vol2pcd(volume, origin, voxel_size, level_set_value=0, quiet=False):
    """
    Converts a binary volume into a point cloud with normals.

    Parameters
    __________
    volume : np.ndarray
        NxMxP 3D binary numpy array
    voxel_size: float
        voxel size
    level_set_value: float
        distance of the level set on which the points are sampled

    Returns
    _______
    open3d.geometry.PointCloud
    """
    volume = volume>0 # variable level ?
    dist = distance_transform_edt(volume)
    mdist = distance_transform_edt(1-volume)
    dist = np.where(dist > 0.5, dist - 0.5, -mdist + 0.5)

    gx, gy, gz = np.gradient(dist)
    gx = gaussian_filter(gx, 1)
    gy = gaussian_filter(gy, 1)
    gz = gaussian_filter(gz, 1)

    on_edge = (dist > -level_set_value) * (dist <= -level_set_value+np.sqrt(3))
    x, y, z = np.nonzero(on_edge)

    if not quiet:
        print("number of points = %d" % len(x))
    pts = np.zeros((0, 3))
    normals = np.zeros((0,3))
    for i in range(len(x)):
        grad = np.array([gx[x[i], y[i], z[i]],
                         gy[x[i], y[i], z[i]],
                         gz[x[i], y[i], z[i]]])
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            grad_normalized = grad / grad_norm
            val = dist[x[i], y[i], z[i]] + level_set_value - np.sqrt(3)/2
            pts = np.vstack([pts, np.array([x[i] - grad_normalized[0] * val,
                                      y[i] - grad_normalized[1] * val,
                                      z[i] - grad_normalized[2] * val])])
            normals = np.vstack([normals, -np.array([grad_normalized[0],
                                                     grad_normalized[1],
                                                     grad_normalized[2]])])

    pts = index2point(pts, origin, voxel_size)
    pcd = PointCloud()
    pcd.points = Vector3dVector(pts)
    pcd.normals = Vector3dVector(normals)
    pcd.normalize_normals()

    return pcd

def crop_point_cloud(point_cloud, bounding_box):
    """
    Crops a point cloud by keeping only points inside bouding box.
    Parameters
    __________
    point_cloud : PointCloud
        input point cloud
    bounding_box : dict
        {"x" : [xmin, xmax], "y" : [ymin, ymax], "z" : [zmin, zmax]}

    Returns
    _______
    PointCloud
    """
    x_bounds = bounding_box['x']
    y_bounds = bounding_box['y']
    z_bounds = bounding_box['z']

    points = np.asarray(point_cloud.points)
    valid_index = ((points[:, 0] > x_bounds[0]) * (points[:, 0] < x_bounds[1]) *
                   (points[:, 1] > y_bounds[0]) * (points[:, 1] < y_bounds[1]) *
                   (points[:, 2] > z_bounds[0]) * (points[:, 2] < z_bounds[1]))

    points = points[valid_index, :]
    cropped_point_cloud = PointCloud()
    cropped_point_cloud.points = Vector3dVector(points)

    if point_cloud.has_normals():
        cropped_point_cloud.normals = Vector3dVector(
            np.asarray(point_cloud.normals)[valid_index, :])

    if point_cloud.has_colors():
        cropped_point_cloud.colors = Vector3dVector(
            np.asarray(point_cloud.colors)[valid_index, :])
    return cropped_point_cloud

def fit_plane_ransac(point_cloud, inliers=0.8, n_iter=100):
    """
    Fits a plane to a point cloud using a Ransac algorithm.
    """
    min_error = np.inf
    argmin_v = None
    argmin_g = None
    coords = np.asarray(point_cloud.points)
    n_inliers = int(np.round(inliers * coords.shape[0]))
    for i in range(n_iter):
        inliers = np.random.choice(range(coords.shape[0]), size=n_inliers)
        inliers_coords = coords[inliers, :]
        G = inliers_coords.mean(axis=0)
        u, s, vh = np.linalg.svd(inliers_coords - G[np.newaxis, :], full_matrices=False)
        if s[2] < min_error:
            argmin_v = vh
            argmin_g = G
            min_error = s[2]
            print("error = %.2f"%s[2])

    X0 = argmin_g # point belonging to the plane
    n = vh[:, 2] # normal vector

    return X0, n

def project_camera_plane(K, rot, tvec, X0, n):
    """
    """

    rot = np.matrix(rot)
    K = np.matrix(K)

    tvec = np.matrix(tvec)
    if tvec.shape[0] == 1:
        tvec = tvec.transpose()

    X0 = np.matrix(X0)
    if X0.shape[0] == 1:
        X0 = X0.transpose()

    n = np.matrix(n)
    if n.shape[0] == 1:
        n = n.transpose()

    f = K[0,0]
    c_x = K[0, 2]
    c_y = K[1, 2]

    # Transform plane in camera frame:
    n_cam, X0_cam = rot*n, rot*X0+ tvec

    # Points in camera frame
    pts = [np.array([-c_x, -c_y, f]), np.array([c_x, -c_y, f]), np.array([-c_x, c_y, f]), np.array([c_x, c_y, f])]

    # Points on target plane in camera frame
    pts_plane = [np.dot(X0_cam.transpose(), n_cam) / np.dot(pt, n_cam) * pt for pt in pts]

    # Points on target plane in world frame
    pts_plane_world = [(rot.transpose()*(pt.transpose() - tvec)).transpose() for pt in pts_plane]

    return np.array(np.vstack(pts_plane_world))

def test_cam_planes(pcd, cameras, images, imgdir, X0=None, n=None, scaling=100):
    w = cameras['1']['width']
    h = cameras['1']['height']

    f,c_x,c_y,_ = cameras['1']['params']
    K = [[f, 0, c_x], [0, f, c_y], [0,0,1]]

    if X0 is None:
        X0, nn = fit_plane_ransac(pcd)
    if n is None:
        n = nn

    rect_lines = [[0,1], [1,2], [2,3], [3,0]]
    rectangles = []
    tri_image = np.array(np.vstack([[0,0], [w, 0], [0, h]]), dtype=np.float32)
    tris = {}
    for k in images.keys():
        if np.random.rand() < 0.9:
            continue
        rot = images[k]['rotmat']
        tvec = images[k]['tvec']
        rect_pts = project_camera_plane(K, rot, tvec, X0, n)
        tri_target = np.array(rect_pts[0:3, :2], dtype=np.float32)
        tris[k] = scaling * tri_target

    xmin = np.min([np.hstack([tri[0, 0], tri[1, 0], tri[2, 0]]) for tri in tris.values()])
    xmax = np.max([np.hstack([tri[0, 0], tri[1, 0], tri[2, 0]]) for tri in tris.values()])
    ymin = np.min([np.hstack([tri[0, 1], tri[1, 1], tri[2, 1]]) for tri in tris.values()])
    ymax = np.max([np.hstack([tri[0, 1], tri[1, 1], tri[2, 1]]) for tri in tris.values()])

    target_image_shape = (int(np.floor(xmax-xmin)), int(np.floor(ymax-ymin)))

    res = np.zeros((target_image_shape[1], target_image_shape[0] , 3), dtype=np.float)

    ks = list(tris.keys())
    ks.sort(key=lambda x: int(x))

    for i, k in enumerate(ks):
        print(k)
        img = imageio.imread(os.path.join(imgdir, images[k]["name"]))
        img = np.array(img, dtype=np.float)
        # img = (img - img.mean()) / img.std()
        tri_target = tris[k]
        tri_target[:,0] -= xmin
        tri_target[:,1] -= ymin
        affine_transform = cv2.getAffineTransform(tri_image, tri_target)
        cv2.warpAffine(img, affine_transform, target_image_shape, dst=res, borderMode=cv2.BORDER_TRANSPARENT)
        # res = np.where(res == 0, img_warped, res)

    # res = np.ma.masked_array(res, res == 0)
    # res = np.ma.median(res, axis=3)
    res = proc2d.rescale_intensity(res,out_range=(0,1))
    return res 




    



