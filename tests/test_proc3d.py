import unittest

import numpy as np
import open3d as o3d
from romiscan import proc3d

class TestProc2D(unittest.TestCase):
    def test_index2point(self):
        indexes = np.zeros((2,3))
        indexes[0,:] = [0,0,0]
        indexes[1,:] = [2,2,2]
        voxel_size = 0.5
        origin = np.array([-0.5, -0.5, -0.5])
        pts = proc3d.index2point(indexes, origin, voxel_size)
        assert(pts.tolist()[0] == origin.tolist())
        assert(pts.tolist()[1] == [0.5, 0.5, 0.5])

    def test_point2index(self):
        origin = np.array([-0.5, -0.5, -0.5])
        pts = np.zeros((2,3))
        pts[0,:] = origin
        pts[1,:] = 0.5
        voxel_size = 0.5
        indexes = proc3d.point2index(pts, origin, voxel_size)
        assert(indexes.tolist()[0] == [0,0,0])
        assert(indexes.tolist()[1] == [2,2,2])

    def test_pcd2mesh(self):
        n_pts = 1000
        np.random.seed(0)
        x = 0.5 - np.random.rand(n_pts, 3)
        x = x / np.linalg.norm(x, axis=1)[:, np.newaxis]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(x)
        pcd.normals = o3d.utility.Vector3dVector(x)
        mesh = proc3d.pcd2mesh(pcd)
        assert(len(mesh.vertices)> 0)

    def test_pcd2vol(self):
        pts = np.zeros((2,3))
        pts[0, :] = 0,0,0
        pts[1, :] = 1,1,1
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        vol, origin = proc3d.pcd2vol(pcd, 1)
        assert(vol.sum() == 2)
        assert(origin.tolist() == [0,0,0])
        assert(vol[0,0,0] == 1)
        assert(vol[1,1,1] == 1)

    def test_skeletonize(self):
        mesh = o3d.io.read_triangle_mesh("testdata/cylinder.ply")
        skel = proc3d.skeletonize(mesh)
        assert(len(skel["points"]) > 0)
        assert(len(skel["lines"]) > 0)

    def test_vol2pcd(self):
        vol = np.zeros((100, 100, 100))
        x,y,z = np.meshgrid(range(-50,50), range(-50,50), range(-50,50))
        vol[x*x+y*y+z*z < 20*20] = 1.0
        pcd = proc3d.vol2pcd(vol, np.array([-50,-50,-50]), 1.0)
        assert(len(pcd.points) > 0)

    def test_crop_point_cloud(self):
        bounding_box = {"x" : [0, 1], "y" : [0, 1], "z" : [0, 1]}
        pts = np.zeros((2,3))
        pts[0, :] = 0.5
        pts[1, :] = -0.5
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        newpcd = proc3d.crop_point_cloud(pcd, bounding_box)
        assert(len(newpcd.points) ==  1)



if __name__ == "__main__":
    unittest.main()
