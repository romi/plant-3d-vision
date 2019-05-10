import os
import tempfile

try:
    from open3d.io import read_point_cloud, write_point_cloud, write_triangle_mesh, read_triangle_mesh
except ImportError:
    from open3d import read_point_cloud, write_point_cloud, write_triangle_mesh, read_triangle_mesh

from lettucethink.db import fsdb


def db_read_point_cloud(file):
    path = fsdb._file_path(file, file.filename)
    return read_point_cloud(path)


def db_write_point_cloud(file, pcd):
    tmpdir = tempfile.TemporaryDirectory()
    pcd_path = os.path.join(tmpdir.name, '%s.ply' % file.id)
    write_point_cloud(pcd_path, pcd)
    file.import_file(pcd_path)


def db_read_triangle_mesh(file):
    path = fsdb._file_path(file, file.filename)
    return read_triangle_mesh(path)


def db_write_triangle_mesh(file, mesh):
    tmpdir = tempfile.TemporaryDirectory()
    mesh_path = os.path.join(tmpdir.name, '%s.ply' % file.id)
    write_triangle_mesh(mesh_path, mesh)
    file.import_file(mesh_path)


def db_read_numpy_array(file):
    path = fsdb._file_path(file, file.filename)
    return np.load(path)


def db_write_numpy_array(file, array):
    tmpdir = tempfile.TemporaryDirectory()
    mesh_path = os.path.join(tmpdir.name, '%s.np' % file.id)
    np.save(mesh_path, array)
    file.import_file(mesh_path)
