import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter

from romiscan import cgal

from open3d.geometry import PointCloud, TriangleMesh
from open3d.utility import Vector3dVector, Vector3iVector

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
