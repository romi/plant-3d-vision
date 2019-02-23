import pyopencl as cl
import os
import numpy as np
from skimage.morphology import binary_dilation


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

with open(os.path.join(os.path.dirname(__file__), 'space_carving.cl')) as f:
    space_carving_program = cl.Program(ctx, f.read()).build()

with open(os.path.join(os.path.dirname(__file__), 'geodesics.cl')) as f:
    geodesics_program = cl.Program(ctx, f.read()).build()


class SpaceCarving():
    def __init__(self, shape, origin, voxel_size):
        self.shape = shape
        self.origin = origin
        self.voxel_size = voxel_size
        self.init_buffers()

    def init_buffers(self):
        self.labels_h = np.zeros(self.shape).astype(np.uint8)
        self.labels_d = cl.Buffer(
            ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.labels_h)

        self.intrinsics_d = cl.Buffer(
            ctx, mf.READ_ONLY, np.zeros(4).astype(np.float32).nbytes)
        self.rot_d = cl.Buffer(
            ctx, mf.READ_ONLY, np.zeros(9).astype(np.float32).nbytes)
        self.tvec_d = cl.Buffer(
            ctx, mf.READ_ONLY, np.zeros(3).astype(np.float32).nbytes)

        self.volinfo_d = cl.Buffer(
            ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.array(
                [*self.origin, self.voxel_size], dtype=np.float32)
        )

        self.shape_d = cl.Buffer(
            ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.array(
                self.shape, dtype=np.int32)
        )

    def process_view(self, intrinsics, rot, tvec, mask):
        intrinsics_h = np.ascontiguousarray(intrinsics).astype(np.float32)
        rot_h = np.ascontiguousarray(rot).astype(np.float32)
        tvec_h = np.ascontiguousarray(tvec).astype(np.float32)
        mask_h = np.ascontiguousarray(mask).astype(np.uint8)

        mask_d = cl.image_from_array(ctx, mask_h, 1)

        cl.enqueue_copy(queue, self.intrinsics_d, intrinsics_h)
        cl.enqueue_copy(queue, self.rot_d, rot_h)
        cl.enqueue_copy(queue, self.tvec_d, tvec_h)

        space_carving_program.check(queue, [np.prod(self.shape)], None, mask_d, self.labels_d,
                                    self.intrinsics_d, self.rot_d,
                                    self.tvec_d, self.volinfo_d, self.shape_d)
        queue.finish()

    def get_labels(self):
        cl.enqueue_copy(queue, self.labels_h, self.labels_d)
        print("%i, %i, %i"%(self.labels_h[0,0,0], self.labels_h[0,0,1], self.labels_h[0,0,2]))
        print(self.shape)
        return self.labels_h
