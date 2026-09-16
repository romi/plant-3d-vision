import unittest
from os.path import abspath
from os.path import join
from pathlib import Path

import numpy as np
import open3d

from plant3dvision import proc3d


class TestProc3D(unittest.TestCase):

    def test_index2point(self):
        indexes = np.zeros((2, 3))
        indexes[0, :] = [0, 0, 0]
        indexes[1, :] = [2, 2, 2]
        voxel_size = 0.5
        origin = np.array([-0.5, -0.5, -0.5])
        pts = proc3d.index2point(indexes, origin, voxel_size)
        assert (pts.tolist()[0] == origin.tolist())
        assert (pts.tolist()[1] == [0.5, 0.5, 0.5])

    def test_point2index(self):
        origin = np.array([-0.5, -0.5, -0.5])
        pts = np.zeros((2, 3))
        pts[0, :] = origin
        pts[1, :] = 0.5
        voxel_size = 0.5
        indexes = proc3d.point2index(pts, origin, voxel_size)
        assert (indexes.tolist()[0] == [0, 0, 0])
        assert (indexes.tolist()[1] == [2, 2, 2])

    def test_pcd2mesh(self):
        n_pts = 1000
        np.random.seed(0)
        x = 0.5 - np.random.rand(n_pts, 3)
        x = x / np.linalg.norm(x, axis=1)[:, np.newaxis]
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(x)
        pcd.normals = open3d.utility.Vector3dVector(x)
        mesh = proc3d.pcd2mesh(pcd)
        assert (len(mesh.vertices) > 0)

    def test_pcd2vol(self):
        pts = np.zeros((2, 3))
        pts[0, :] = 0, 0, 0
        pts[1, :] = 1, 1, 1
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pts)
        vol, origin = proc3d.pcd2vol(pcd, 1)
        assert (vol.sum() == 2)
        assert (origin.tolist() == [0, 0, 0])
        assert (vol[0, 0, 0] == 1)
        assert (vol[1, 1, 1] == 1)

    def test_skeletonize(self):
        parent_dir = Path(__file__).resolve().parents[1]
        cylinder_mesh_file = abspath(join(parent_dir, "testdata", "cylinder.ply"))

        mesh = open3d.io.read_triangle_mesh(cylinder_mesh_file)
        skel = proc3d.mesh_to_skeleton(mesh)
        assert (len(skel["points"]) > 0)
        assert (len(skel["lines"]) > 0)

    def test_vol2pcd(self):
        vol = np.zeros((100, 100, 100))
        x, y, z = np.meshgrid(range(-50, 50), range(-50, 50), range(-50, 50))
        vol[x * x + y * y + z * z < 20 * 20] = 1.0
        pcd = proc3d.vol2pcd(vol, np.array([-50, -50, -50]), 1.0)
        assert (len(pcd.points) > 0)

    def test_crop_point_cloud(self):
        bounding_box = {"x": [0, 1], "y": [0, 1], "z": [0, 1]}
        pts = np.zeros((2, 3))
        pts[0, :] = 0.5
        pts[1, :] = -0.5
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pts)
        newpcd = proc3d.crop_point_cloud(pcd, bounding_box)
        assert (len(newpcd.points) == 1)

    @staticmethod
    def _gmrf_volume():
        # Non-uniform volume (a block + a little noise) so smoothing changes values.
        rng = np.random.default_rng(0)
        vol = np.zeros((16, 16, 16))
        vol[6:10, 6:10, 6:10] = 1.0
        vol = vol + 0.01 * rng.normal(size=vol.shape)
        return vol

    def test_smooth_volume_gmrf(self):
        vol = self._gmrf_volume()
        out = proc3d.smooth_volume_gmrf(vol, lam=1.0)
        assert (out.shape == vol.shape)
        assert (not np.array_equal(out, vol))

    def test_smooth_volume_gmrf_lam0_disables_smoothing(self):
        vol = self._gmrf_volume()
        out = proc3d.smooth_volume_gmrf(vol, lam=0.0)
        assert (np.array_equal(out, vol))

    def test_smooth_volume_gmrf_linearop(self):
        vol = self._gmrf_volume()
        out = proc3d.smooth_volume_gmrf_linearop(vol, lam=1.0)
        assert (out.shape == vol.shape)
        assert (not np.array_equal(out, vol))

    def test_smooth_volume_gmrf_linearop_lam0_disables_smoothing(self):
        vol = self._gmrf_volume()
        out = proc3d.smooth_volume_gmrf_linearop(vol, lam=0.0)
        assert (np.array_equal(out, vol))

    def test_smooth_volume_gmrf_linearop_matches_explicit(self):
        # Both methods solve the same (I + λL)x = b system.
        vol = self._gmrf_volume()
        ref = proc3d.smooth_volume_gmrf(vol, lam=1.0)
        out = proc3d.smooth_volume_gmrf_linearop(vol, lam=1.0)
        assert (np.allclose(ref, out, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
