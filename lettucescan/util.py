from lettucethink import fsdb
import tempfile
import os

def db_read_point_cloud(file):
    path = fsdb._file_path(file, file.filename)
    return open3d.read_point_cloud(path)

def db_write_point_cloud(file, pcd):
    tmpdir = tempfile.TemporaryDirectory()
    pcd_path = os.path.join(tmpdir.name, 'pcd.ply')
    open3d.write_point_cloud(pcd_path, pcd)
    file.import_file(pcd_path)

def db_read_triangle_mesh(file):
    path = fsdb._file_path(file, file.filename)
    return open3d.read_triangle_mesh(path)

def db_write_triangle_mesh(file, mesh):
        tmpdir = tempfile.TemporaryDirectory()
        mesh_path = os.path.join(tmpdir.name, 'mesh.ply')
        open3d.write_triangle_mesh(mesh_path, mesh)
        file.import_file(mesh_path)
