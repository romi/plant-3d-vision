#!/usr/bin/env python3
"""
Unit tests for the Bayesian space carving (CUDA) backprojection and the GMRF
volume post-processing smoothing, including a parity check against the Julia
``ROMIVoxels.jl`` reference logic.
"""

import unittest

import numpy as np

from plant3dvision.proc3d import smooth_volume_gmrf


def reference_voxel_voting(shape, origin, voxel_size, frames, masks, prior_prob, tpr, fpr):
    """Pure-NumPy port of the Julia ``voxel_voting`` log-odds accumulation.

    Mirrors the CUDA kernel geometry (world point = origin + index*voxel_size,
    pinhole projection with no radial distortion) so the test checks the
    log-odds formula and per-view accumulation, independent of the CUDA path.
    """
    lod = np.full(shape, np.log(prior_prob / (1.0 - prior_prob)))
    occ = np.log(tpr / fpr)
    emp = np.log((1.0 - tpr) / (1.0 - fpr))
    nx, ny, nz = shape
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                pt = np.array([origin[0] + x * voxel_size,
                               origin[1] + y * voxel_size,
                               origin[2] + z * voxel_size])
                for intr, rot, tvec, mask in zip(frames["intrinsics"], frames["rotmat"], frames["tvec"], masks):
                    pc = rot @ pt + tvec
                    if pc[2] <= 0:
                        continue
                    u = int(pc[0] / pc[2] * intr[0] + intr[2])
                    v = int(pc[1] / pc[2] * intr[1] + intr[3])
                    if 0 <= u < mask.shape[1] and 0 <= v < mask.shape[0]:
                        lod[x, y, z] += occ if mask[v, u] > 0 else emp
    return lod


class TestBayesianBackprojectionCUDA(unittest.TestCase):
    """Parity check of the CUDA log-odds volume against a NumPy reference."""

    def _skip_no_cuda(self):
        try:
            from plant3dvision.voxels_bayes_cuda import BayesianBackprojection
            return BayesianBackprojection
        except Exception as e:
            self.skipTest(f"CUDA test failed: {e}")

    def test_logodds_matches_reference(self):
        BayesianBackprojection = self._skip_no_cuda()
        shape = [4, 4, 4]
        origin = [0.0, 0.0, 0.0]
        voxel_size = 1.0
        prior_prob, tpr, fpr = 0.05, 0.95, 0.1

        bp = BayesianBackprojection(shape, origin, voxel_size, prior_prob=prior_prob,
                                    tpr=tpr, fpr=fpr)

        # Two synthetic views with distinct cameras.
        masks = []
        frames = {"intrinsics": [], "rotmat": [], "tvec": []}
        for tz, fg_pixel in ((10.0, (5, 5)), (8.0, (6, 4))):
            w, h = 10, 10
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[fg_pixel] = 255
            masks.append(mask)
            frames["intrinsics"].append(np.array([10.0, 10.0, 5.0, 5.0], dtype=np.float32))
            frames["rotmat"].append(np.eye(3, dtype=np.float32))
            frames["tvec"].append(np.array([0.0, 0.0, tz], dtype=np.float32))

        fs = {f"m{i}": m for i, m in enumerate(masks)}
        md = {f"m{i}": {"intrinsics": frames["intrinsics"][i],
                        "rotmat": frames["rotmat"][i],
                        "tvec": frames["tvec"][i]} for i in range(len(masks))}

        vol = bp.process_fileset(fs, md)
        ref = reference_voxel_voting(shape, origin, voxel_size, frames, masks,
                                     prior_prob, tpr, fpr)
        np.testing.assert_allclose(vol, ref, atol=1e-4)


class TestGMRFSmoothing(unittest.TestCase):
    """Parity/consistency checks for the GMRF MAP smoothing port."""

    def test_lambda_zero_returns_copy(self):
        v = np.random.rand(4, 5, 6)
        out = smooth_volume_gmrf(v, tau=10.0, lam=0.0)
        self.assertTrue(np.array_equal(out, v))

    def test_matches_dense_solution(self):
        # Verify the sparse-matrix CG solve equals a direct dense solve of (I+λL)x = b.
        from scipy.sparse import identity
        from scipy.sparse.linalg import cg
        from plant3dvision.proc3d import _gmrf_laplacian

        rng = np.random.default_rng(0)
        D = rng.random((3, 3, 3))
        tau, lam = 2.0, 0.5
        N = D.size
        L = _gmrf_laplacian(D, tau)
        A = identity(N, format='csr') + lam * L
        b = D.ravel()
        x, _ = cg(A, b, x0=b, rtol=1e-9)
        x_dense = np.linalg.solve((np.eye(N) + lam * L.toarray()), b)
        np.testing.assert_allclose(x, x_dense, atol=1e-6)
        # And the public function agrees.
        out = smooth_volume_gmrf(D, tau=tau, lam=lam, tol=1e-9)
        np.testing.assert_allclose(out.ravel(), x_dense, atol=1e-5)

    def test_flat_volume_stays_flat(self):
        v = np.full((4, 4, 4), 2.0)
        out = smooth_volume_gmrf(v, tau=10.0, lam=1.0)
        np.testing.assert_allclose(out, 2.0, atol=1e-8)

    def test_edge_preserved_for_small_tau(self):
        # A sharp step should be preserved (Welsch weight ~ 0 across the edge).
        v = np.zeros((6, 6, 6))
        v[3:, :, :] = 10.0
        out = smooth_volume_gmrf(v, tau=0.1, lam=5.0)
        # Boundary voxels on each side keep their value (edge not blurred).
        self.assertAlmostEqual(out[0, 3, 3], 0.0, delta=0.05)
        self.assertAlmostEqual(out[5, 3, 3], 10.0, delta=0.05)


if __name__ == '__main__':
    unittest.main()
