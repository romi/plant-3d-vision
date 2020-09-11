from romiscan.tasks.colmap import *
from romiscan.tasks.proc2d import *
from romiscan.tasks.proc3d import *
from romiscan.tasks.arabidopsis import *

logger = logging.getLogger('__name__')


class Visualization(RomiTask):
    """Prepares files for visualization
    """
    upstream_task = None

    upstream_point_cloud = luigi.TaskParameter(default=PointCloud)
    upstream_mesh = luigi.TaskParameter(default=TriangleMesh)
    upstream_colmap = luigi.TaskParameter(default=Colmap)
    upstream_angles = luigi.TaskParameter(default=AnglesAndInternodes)
    upstream_skeleton = luigi.TaskParameter(default=CurveSkeleton)
    upstream_images = luigi.TaskParameter(default=Undistorted)

    max_image_size = luigi.IntParameter()
    max_point_cloud_size = luigi.IntParameter()
    thumbnail_size = luigi.IntParameter()

    def __init__(self):
        super().__init__()
        self.task_id = "Visualization"

    def requires(self):
        return []

    def run(self):
        import tempfile
        import shutil
        import os
        try:
            from open3d import open3d
        except:
            import open3d
        from skimage.transform import resize

        def resize_to_max(img, max_size):
            i = np.argmax(img.shape[0:2])
            if img.shape[i] <= max_size:
                return img
            if i == 0:
                new_shape = [max_size, int(max_size * img.shape[1] / img.shape[0])]
            else:
                new_shape = [int(max_size * img.shape[0] / img.shape[1]), max_size]
            return resize(img, new_shape)

        output_fileset = self.output().get()
        files_metadata = {
            "zip": None,
            "angles": None,
            "skeleton": None,
            "mesh": None,
            "point_cloud": None,
            "images": None,
            "poses": None,
            "thumbnails": None
        }

        # POSES
        colmap_fileset = self.upstream_colmap().output().get()
        images = io.read_json(colmap_fileset.get_file(COLMAP_IMAGES_ID))
        camera = io.read_json(colmap_fileset.get_file(COLMAP_CAMERAS_ID))
        camera["bounding_box"] = colmap_fileset.get_metadata("bounding_box")
        f = output_fileset.get_file(COLMAP_IMAGES_ID, create=True)
        f_cam = output_fileset.get_file(COLMAP_CAMERAS_ID, create=True)
        io.write_json(f, images)
        io.write_json(f_cam, camera)
        files_metadata["poses"] = f.id
        files_metadata["camera"] = f_cam.id

        # ZIP
        scan = self.output().scan
        basedir = scan.db.basedir
        logger.debug("basedir = %s" % basedir)
        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.make_archive(os.path.join(tmpdir, "scan"), "zip",
                                os.path.join(basedir, scan.id))
            f = output_fileset.get_file('scan', create=True)
            f.import_file(os.path.join(tmpdir, 'scan.zip'))
        files_metadata["zip"] = scan.id

        # ANGLES
        if self.upstream_angles().complete():
            angles_file = self.upstream_angles().output_file()
            f = output_fileset.create_file(angles_file.id)
            io.write_json(f, io.read_json(angles_file))
            files_metadata["angles"] = angles_file.id

        # SKELETON
        if self.upstream_skeleton().complete():
            skeleton_file = self.upstream_skeleton().output_file()
            f = output_fileset.create_file(skeleton_file.id)
            io.write_json(f, io.read_json(skeleton_file))
            files_metadata["skeleton"] = skeleton_file.id

        # MESH
        if self.upstream_mesh().complete():
            mesh_file = self.upstream_mesh().output_file()
            f = output_fileset.create_file(mesh_file.id)
            io.write_triangle_mesh(f, io.read_triangle_mesh(mesh_file))
            files_metadata["mesh"] = mesh_file.id

        # PCD
        if self.upstream_point_cloud().complete():
            point_cloud_file = self.upstream_point_cloud().output_file()
            point_cloud = io.read_point_cloud(point_cloud_file)
            f = output_fileset.create_file(point_cloud_file.id)
            if len(point_cloud.points) < self.max_point_cloud_size:
                point_cloud_lowres = point_cloud
            else:
                try:
                    point_cloud_lowres = open3d.geometry.uniform_down_sample(point_cloud, len(point_cloud.points) // self.max_point_cloud_size + 1)
                except:
                    point_cloud_lowres = point_cloud.voxel_down_sample(len(point_cloud.points) // self.max_point_cloud_size + 1)
            io.write_point_cloud(f, point_cloud)
            files_metadata["point_cloud"] = point_cloud_file.id

        # IMAGES
        images_fileset = self.upstream_images().output().get()
        files_metadata["images"] = []
        files_metadata["thumbnails"] = []

        for img in images_fileset.get_files():
            data = io.read_image(img)
            # remove alpha channel
            if data.shape[2] == 4:
                data = data[:, :, :3]
            image = resize_to_max(data, self.max_image_size)
            thumbnail = resize_to_max(data, self.thumbnail_size)

            image_id = "image_%s" % img.id
            thumbnail_id = "thumbnail_%s" % img.id

            f = output_fileset.create_file(image_id)
            io.write_image(f, image)
            f.set_metadata("image_id", img.id)

            f = output_fileset.create_file(thumbnail_id)
            io.write_image(f, thumbnail)
            f.set_metadata("image_id", img.id)

            files_metadata["images"].append(image_id)
            files_metadata["thumbnails"].append(thumbnail_id)

        # MEASURES
        measures = scan.get_measures()
        f_measures = output_fileset.create_file("measures")
        io.write_json(f_measures, measures)
        files_metadata["measures"] = "measures"

        # DESCRIPTION OF FILES
        output_fileset.set_metadata("files", files_metadata)
