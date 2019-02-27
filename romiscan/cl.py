import pyopencl as cl
import os
import numpy as np
from lettucescan.pcd import point2index
from skimage.morphology import binary_dilation


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

with open(os.path.join(os.path.dirname(__file__), 'kernels/backprojection.c')) as f:
    space_carving_program = cl.Program(ctx, f.read()).build()

with open(os.path.join(os.path.dirname(__file__), 'kernels/geodesics.c')) as f:
    geodesics_program = cl.Program(ctx, f.read()).build()

class Backprojection():
    def __init__(self, shape, origin, voxel_size, type="space_carving", default_value=0):
        self.shape = shape
        self.origin = origin
        self.voxel_size = voxel_size
        self.default_value = default_value

        self.type = type
        if type == "carving":
            self.dtype = np.uint8
            self.kernel = backprojection_kernels.carve
        elif type == "averaging":
            self.dtype = np.float32
            self.kernel = backprojection_kernels.average

        self.init_buffers()


    def init_buffers(self):
        self.values_h = self.default_value * np.ones(self.shape).astype(self.dtype)

        self.values_d = cl.Buffer(
            ocl_ctx, ocl_mf.READ_WRITE | ocl_mf.COPY_HOST_PTR, hostbuf=self.values_h)

        self.intrinsics_d = cl.Buffer(
            ocl_ctx, ocl_mf.READ_ONLY, np.zeros(4).astype(np.float32).nbytes)
        self.rot_d = cl.Buffer(
            ocl_ctx, ocl_mf.READ_ONLY, np.zeros(9).astype(np.float32).nbytes)
        self.tvec_d = cl.Buffer(
            ocl_ctx, ocl_mf.READ_ONLY, np.zeros(3).astype(np.float32).nbytes)

        self.volinfo_d = cl.Buffer(
            ocl_ctx, ocl_mf.READ_WRITE | ocl_mf.COPY_HOST_PTR, hostbuf=np.array(
                [*self.origin, self.voxel_size], dtype=np.float32)
        )

        self.shape_d = cl.Buffer(
            ocl_ctx, ocl_mf.READ_WRITE | ocl_mf.COPY_HOST_PTR, hostbuf=np.array(
                self.shape, dtype=np.int32)
        )

    def process_view(self, intrinsics, rot, tvec, mask):
        intrinsics_h = np.ascontiguousarray(intrinsics).astype(np.float32)
        rot_h = np.ascontiguousarray(rot).astype(np.float32)
        tvec_h = np.ascontiguousarray(tvec).astype(np.float32)
        mask_h = np.ascontiguousarray(mask).astype(self.dtype)

        mask_d = cl.image_from_array(ocl_ctx, mask_h, 1)

        cl.enqueue_copy(ocl_queue, self.intrinsics_d, intrinsics_h)
        cl.enqueue_copy(ocl_queue, self.rot_d, rot_h)
        cl.enqueue_copy(ocl_queue, self.tvec_d, tvec_h)

        self.kernel(ocl_queue, [np.prod(self.shape)], None, mask_d, self.values_d,
            self.intrinsics_d, self.rot_d,
            self.tvec_d, self.volinfo_d, self.shape_d)
        ocl_queue.finish()

    def values(self):
        cl.enqueue_copy(ocl_queue, self.values_h, self.values_d)
        return self.values_h


class Geodesics():
    def __init__(self):
        return

    def compute_geodesics(self, values, origin, voxel_size, flow, tips, max_iters, step_size):
        shape = values.shape
        tips = point2index(tips, origin, voxel_size)

        gx_d = cl.image_from_array(ctx, np.ascontiguousarray(flow[:, :, :, 0]).astype(np.float32), 1)
        gy_d = cl.image_from_array(ctx, np.ascontiguousarray(flow[:, :, :, 1]).astype(np.float32), 1)
        gz_d = cl.image_from_array(ctx, np.ascontiguousarray(flow[:, :, :, 2]).astype(np.float32), 1)

        values_h = np.ascontiguousarray(values).astype(np.float32)
        values_d = cl.image_from_array(ctx, values_h, 1)

        points_h = np.ascontiguousarray(tips).astype(np.float32)
        points_d = cl.Buffer(ctx, mf.READ_WRITE |
                             mf.COPY_HOST_PTR, hostbuf=points_h)

        labels_h = np.ones(tips.shape, dtype=np.uint8)
        labels_d = cl.Buffer(ctx, mf.READ_WRITE |
                             mf.COPY_HOST_PTR, hostbuf=labels_h)

        votes_h = np.zeros(shape, dtype=np.int32)
        votes_d = cl.Buffer(ctx, mf.READ_WRITE |
                            mf.COPY_HOST_PTR, hostbuf=votes_h.ravel())

        points_remain_h = np.asarray([True], dtype=np.int32)
        points_remain_d = cl.Buffer(
            ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=points_remain_h)

        shape_h = np.array(shape, dtype=np.int32)
        shape_d = cl.Buffer(ctx, mf.READ_WRITE |
                            mf.COPY_HOST_PTR, hostbuf=shape_h)
        idx = []

        for i in range(max_iters):
            kernel = geodesics_program.geodesic
            kernel.set_scalar_arg_dtypes([None, None, None, None, None, None,
                                          None, None, None, np.float32])
            kernel(queue, (tips.shape[0],), None,
                                       gx_d, gy_d, gz_d, values_d,
                                       votes_d, points_d, labels_d, points_remain_d,
                                       shape_d, np.float32(step_size))
            cl.enqueue_copy(queue, points_remain_h, points_remain_d)
            queue.finish()
            if not points_remain_h[0]:
                break
        cl.enqueue_copy(queue, votes_h.ravel(), votes_d)
        queue.finish()
        return votes_h
