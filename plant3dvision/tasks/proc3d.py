import luigi
import numpy as np
import open3d as o3d

from romitask import RomiTask
from plantdb import io
from plant3dvision import proc3d
from plant3dvision.log import logger
from plant3dvision.tasks import config
from plant3dvision.tasks.cl import Voxels
from plant3dvision.tasks.colmap import Colmap
from plant3dvision.tasks.proc2d import Segmentation2D

try:
    from hdbscan import HDBSCAN
except ImportError:
    logger.warning("HDBSCAN clustering method is not available!")

class PointCloud(RomiTask):
    """ Computes a point cloud from volumetric voxel data (either single or multiclass).

    Module: plant3dvision.tasks.proc3d
    Default upstream tasks: Voxels
    Upstream task format: npz file with as many 3D array as classes
    Output task format: single point cloud in ply. Metadata may include label name if multiclass.

    """
    upstream_task = luigi.TaskParameter(default=Voxels)
    level_set_value = luigi.FloatParameter(default=1.0)
    log = luigi.BoolParameter(default=False)
    background_prior = luigi.FloatParameter(default=1.0)
    min_contrast = luigi.FloatParameter(default=10.0)
    min_score = luigi.FloatParameter(default=0.2)

    def run(self):
        ifile = self.input_file()
        try:
            voxels = io.read_npz(ifile)
            if (len(voxels.keys()) == 1):
                multiclass = False
                voxels = voxels[list(voxels.keys())[0]]
            else:
                multiclass = True
        except:
            voxels = io.read_volume(ifile)
            multiclass = False
        if multiclass:
            l = list(voxels.keys())
            # background_idx = l.index("background")
            # l.remove("background")
            res = np.zeros((*voxels[l[0]].shape, len(l)))
            for i in range(len(l)):
                res[:, :, :, i] = voxels[l[i]]
            for i in range(len(l)):
                if l[i] == 'background':
                    res[:, :, :, i] *= self.background_prior

            # bg = voxels["background"] > voxels["background"].max() - 10

            res_idx = np.argmax(res, axis=3)
            # res_value = np.amax(res, axis=3)

            # threshold= np.quantile(res_value.flatten(), 0.99)
            # res_idx[res_value < threshold] = background_idx # low scores belong to background

            pcd = o3d.geometry.PointCloud()
            origin = np.array(ifile.get_metadata('origin'))

            voxel_size = float(ifile.get_metadata('voxel_size'))
            point_labels = []
            colors = config.PointCloudColorConfig().colors

            for i in range(len(l)):
                logger.debug(f"label = {l[i]}")
                if l[i] != 'background':
                    pred_no_c = np.copy(res)
                    pred_no_c = np.max(np.delete(res, i, axis=3), axis=3)
                    pred_c = res[:, :, :, i]
                    pred_c = (res_idx == i)
                    if self.min_contrast > 1.0:
                        pred_c *= (pred_c > (self.min_contrast * pred_no_c))
                    pred_c *= (pred_c > self.min_score)

                    out = proc3d.vol2pcd_p(pred_c, origin, voxel_size, self.level_set_value)
                    color = np.zeros((len(out.points), 3))
                    if l[i] in colors:
                        color[:] = np.asarray(colors[l[i]])
                    else:
                        color[:] = np.random.rand(3)
                    color = o3d.utility.Vector3dVector(color)
                    out.colors = color
                    pcd = pcd + out
                    point_labels = point_labels + [l[i]] * len(out.points)

            io.write_point_cloud(self.output_file(), pcd)
            self.output_file().set_metadata({'labels': point_labels})

        else:
            origin = np.array(ifile.get_metadata('origin'))
            voxel_size = float(ifile.get_metadata('voxel_size'))
            out = proc3d.vol2pcd_p(voxels, origin, voxel_size, self.level_set_value)
            io.write_point_cloud(self.output_file(), out)
            self.output_file().set_metadata({'voxel_size': voxel_size})


class SelectLongestStem(RomiTask):
    """Select the longest stem from the point cloud.

    Obtained point-clouds often suffer from reconstruction aberrations such as a broken stem or other artifacts.

    We first cluster the point-cloud using a density based approach to detect the groups of points.
    Then we use the inertia axes to select the longest stem under the hypothesis that it will:
      - be made of a lots of point;
      - have a first axis with a norm much larger than the second.

    Hence this should work well with elongated structure such as arabidopsis plants!

    Attributes
    ----------
    upstream_task : {PointCloud, SegmentedPointCloud}
        The upstream task, should return a PointCloud object.
    min_pts : int
        The minimum number of point to consider the cluster as a potential candidate for the longest stem
    clustering_method : {"dbscan", "hdbscan"}
        The clustering method to use, see references.
    eps : float
        The density parameter, used by DBSCAN clustering method.
    min_points : int
        The number of samples in a neighbourhood for a point to be considered a core point.
    min_cluster_size : int
        The minimum size of clusters; single linkage splits that contain fewer points than this will be considered
        points “falling out” of a cluster rather than a cluster splitting into two new clusters.

    References
    ----------
    https://hdbscan.readthedocs.io/en/latest/api.html
    http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#DBSCAN-clustering

    """
    upstream_task = luigi.TaskParameter(default=PointCloud)  # should also accept 'SegmentedPointCloud'
    min_pts = luigi.IntParameter(default=500)  # minimum number of points in a cluster to consider the cluster with criterion

    clustering_method = luigi.Parameter(default="dbscan")  # ["dbscan", "hdbscan"]
    eps = luigi.FloatParameter(default=2.0)  # dbscan
    min_points = luigi.IntParameter(default=5)  # dbscan (equivalent to 'min_sample' in hdbscan)
    min_cluster_size = luigi.IntParameter(default=5)  # hdbscan

    def run(self):
        # Read the (labelled)point-cloud from the `upstream_task`
        pcd = io.read_point_cloud(self.input_file())
        pcd_arr = np.asarray(pcd.points)

        # - Clustering of the whole point-cloud:
        # DBSCAN clustering:
        if self.clustering_method == "dbscan":
            clustered_arr = np.array(pcd.cluster_dbscan(eps=self.eps, min_points=self.min_points, print_progress=True))
        # HDBSCAN clustering:
        elif self.clustering_method == "hdbscan":
            clusterer = HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_points, core_dist_n_jobs=-1)
            clustered_arr = np.array(clusterer.fit(pcd_arr).labels_)

        # List of cluster ids
        ids = np.unique(clustered_arr).tolist()
        if -1 in ids:
            ids.remove(-1)  # outliers id!
        logger.info(f"Got {len(ids)} clusters in point-could!")

        # - Compute the covariance matrices and number of points in each cluster:
        eig_vals = {}
        n_pts = {}
        for idx in ids:
            sel_pts = pcd_arr[clustered_arr == idx]
            n_pts[idx] = len(sel_pts)
            logger.info(f"Got {n_pts[idx]} points in cluster {idx}!")
            cov_mat = np.cov(sel_pts, rowvar=False)  # covariance matrix from subset of points
            eig_val, _ = np.linalg.eig(cov_mat)  # eigen value to get inertia matrix
            logger.debug(f"The inertia axes norms are: {eig_val}")
            eig_vals[idx] = eig_val

        def criterion(n_pts, eig):
            """Selection criterion is the ratio of the two largest components multiplied by the number of points."""
            return n_pts * eig[0]/eig[1]

        # - Use the criterion to find the cluster of points
        # Compute it for the first cluster:
        longest_stem_idx = 0
        coef = criterion(n_pts[longest_stem_idx], eig_vals[longest_stem_idx])
        # Iterate and select the one with the largest criteria value
        for idx in ids[1:]:
            if n_pts[idx] < self.min_pts:
                continue  #skip if number of point is below defined threshold
            new_coef = criterion(n_pts[idx], eig_vals[idx])
            if new_coef > coef:
                longest_stem_idx = idx
        logger.info(f"Selected cluster {longest_stem_idx} as the longest stem point-cloud!")

        # - Get the index of points matching the longest stem index
        idx_mask = np.where(clustered_arr == longest_stem_idx)[0]
        stem_pcd = pcd.select_by_index(list(idx_mask))
        io.write_point_cloud(self.output_file(), stem_pcd)
        # - Get a bounding-box
        bbox = stem_pcd.get_axis_aligned_bounding_box()
        xyz_bbox = np.array([bbox.get_min_bound(), bbox.get_max_bound()]).T
        extent = bbox.get_extent()
        margin = min(0.05 * extent)  # a 5% margin on bounding-box extent
        xyz_bbox += np.array([-margin, margin])
        logger.info(f"New region of interest (XYZ bounds): {xyz_bbox}")
        # TODO: export bounding-box to metadata !!!!


class SegmentedPointCloud(RomiTask):
    """Segments an existing point cloud using 2D pictures."""
    upstream_task = luigi.TaskParameter(default=Colmap)
    upstream_segmentation = luigi.TaskParameter(default=Segmentation2D)
    use_colmap_poses = luigi.BoolParameter(default=True)

    def requires(self):
        return [self.upstream_task(), self.upstream_segmentation()]

    def load_point_cloud(self):
        try:
            x = self.requires()[0].output().get().get_file("dense")
            return io.read_point_cloud(x)
        except:
            x = self.requires()[0].output().get().get_files()[0]
            return io.read_point_cloud(x)

    def is_in_pict(self, px, shape):
        return px[0] >= 0 and px[0] < shape[1] and px[1] >= 0 and px[1] < shape[0]

    def run(self):
        fs = self.upstream_segmentation().output().get()
        pcd = self.load_point_cloud()
        pts = np.asarray(pcd.points)
        ifile = self.input_file()

        labels = set()
        for fi in fs.get_files():
            label = fi.get_metadata('channel')
            if label is not None:
                labels.add(label)
        labels = list(labels)
        labels.remove('background')
        if 'rgb' in labels:
            labels.remove('rgb')

        scores = np.zeros((len(labels), len(pts)))

        for fi in fs.get_files():
            label = fi.get_metadata("channel")
            if label not in labels:
                continue

            if self.use_colmap_poses:
                camera = fi.get_metadata("colmap_camera")
            else:
                camera = fi.get_metadata("camera")

            if camera is None:
                logger.warning(
                    "Could not get camera pose for view, skipping...")
                continue

            rotmat = np.array(camera["rotmat"])
            tvec = np.array(camera["tvec"])

            intrinsics = camera["camera_model"]["params"]
            K = np.array([[intrinsics[0], 0, intrinsics[2]],
                          [0, intrinsics[1], intrinsics[3]], [0, 0, 1]])
            pixels = np.asarray(
                proc3d.backproject_points(pts, K, rotmat, tvec) + 0.5,
                dtype=int)

            label_idx = labels.index(label)
            mask = io.read_image(fi)
            for i, px in enumerate(pixels):
                if self.is_in_pict(px, mask.shape):
                    scores[label_idx, i] += mask[px[1], px[0]]

        pts_labels = np.argmax(scores, axis=0).flatten()
        logger.critical(f"Processed following labels: {labels}")

        colors = config.PointCloudColorConfig().colors
        logger.critical(f"Associated colors: {colors}")

        color_array = np.zeros((len(pts), 3))
        point_labels = [""] * len(pts)
        for i in range(len(labels)):
            nlab_pts = (pts_labels == i).sum()
            logger.critical(f"Number of points associated to label '{labels[i]}': {nlab_pts}")
            if labels[i] in colors:
                color_array[pts_labels == i, :] = np.asarray(colors[labels[i]])
            else:
                color_array[pts_labels == i, :] = np.random.rand(3)
            l = np.nonzero(pts_labels == i)[0].tolist()
            for u in l:
                point_labels[u] = labels[i]
        pcd.colors = o3d.utility.Vector3dVector(color_array)
        out = self.output_file()
        io.write_point_cloud(out, pcd)
        out.set_metadata("labels", point_labels)


class TriangleMesh(RomiTask):
    """ Triangulates input point cloud.

    Currently, ignores class data and needs only one connected component.

    Module: plant3dvision.tasks.proc3d
    Default upstream tasks: PointCloud
    Upstream task format: ply file
    Output task format: ply triangle mesh file

    """
    upstream_task = luigi.TaskParameter(default=PointCloud)
    library = luigi.Parameter(default="open3d")  # ["cgal", "open3d"]
    filtering = luigi.Parameter(default="most connected triangles")  # ["", "most connected triangles", "largest connected triangles", "dbscan point-cloud"]

    def run(self):
        from plant3dvision import proc3d
        point_cloud = io.read_point_cloud(self.input_file())

        # TODO: Add DBSCAN clustering method to filter the point-cloud prior to meshing
        if self.filtering == "dbscan point-cloud":
            raise NotImplementedError("Coming soon!")

        if self.library == "cgal":
            out = proc3d.pcd2mesh(point_cloud)
        elif self.library == "open3d":
            out, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud)

        # If a filtering method based on connected triangles is required, perform it:
        if "connected triangle" in self.filtering:
            triangle_clusters, cluster_n_triangles, cluster_area = out.cluster_connected_triangles()
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            cluster_area = np.asarray(cluster_area)
            n_cluster = cluster_n_triangles.shape[0]
            logger.info(f"Found {n_cluster} clusters of triangles!")
            logger.info(f"Area of each cluster: {cluster_area}.")
            logger.info(f"Number of triangles in each cluster: {cluster_n_triangles}.")
        if self.filtering == "most connected triangles":
            # Get the index of the largest cluster in the number of triangles
            largest_cluster_idx = cluster_n_triangles.argmax()
            logger.info(f"Cluster #{largest_cluster_idx} was selected!")
            # Creates a mask of triangle to remove and filter them out of the mesh:
            triangles_to_remove = triangle_clusters != largest_cluster_idx
            out.remove_triangles_by_mask(triangles_to_remove)
        elif self.filtering == "largest connected triangles":
            # Get the index of the largest cluster in the number of total area
            largest_cluster_idx = cluster_area.argmax()
            logger.info(f"Cluster #{largest_cluster_idx} was selected!")
            # Creates a mask of triangle to remove and filter them out of the mesh:
            triangles_to_remove = triangle_clusters != largest_cluster_idx
            out.remove_triangles_by_mask(triangles_to_remove)

        io.write_triangle_mesh(self.output_file(), out)


class ClusteredMesh(RomiTask):
    upstream_task = luigi.TaskParameter(default=SegmentedPointCloud)
    min_vol = luigi.FloatParameter(default=1.0)
    min_length = luigi.FloatParameter(default=10.0)

    def run(self):
        x = io.read_point_cloud(self.input_file())
        all_points = np.asarray(x.points)
        all_normals = np.asarray(x.normals)
        all_colors = np.asarray(x.colors)

        # Get the list of semantic label ('flower', 'fruit', ...) attached to each points of the point cloud
        labels = self.input_file().get_metadata("labels")
        output_fileset = self.output().get()
        # Loop on the unique set of labels:
        for l in set(labels):
            pcd = o3d.geometry.PointCloud()
            # Get the index of points matching the semantic label
            idx = [i for i in range(len(labels)) if labels[i] == l]
            # Select points, normals & colors for those point (to reconstruct a point cloud)
            points = all_points[idx, :]
            normals = all_normals[idx, :]
            colors = all_colors[idx, :]
            # Skip point cloud reconstruction if no points corresponding to label
            if len(points) == 0:
                logger.critical(f"No points found for label: '{l}'")
                continue
            # Reconstruct colored point cloud with normals:
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            # Mesh the point cloud (built with the points corresponding to the label)
            t, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)
            t.compute_adjacency_list()
            k, cc, _ = t.cluster_connected_triangles()
            k = np.asarray(k)
            tri_np = np.asarray(t.triangles)
            for j in range(len(cc)):
                newt = o3d.geometry.TriangleMesh(t.vertices,
                                                 o3d.utility.Vector3iVector(tri_np[k == j, :]))
                newt.vertex_colors = t.vertex_colors
                newt.remove_unreferenced_vertices()

                f = output_fileset.create_file("%s_%03d" % (l, j))
                io.write_triangle_mesh(f, newt)
                f.set_metadata("label", l)


class OrganSegmentation(RomiTask):
    """Organ detection using DBSCAN clustering on the SegmentedPointCloud.

    This is done for each semantic label ('flower', 'fruit', ...) of the labelled point cloud,
    except for the stem as it is considered to be one organ.
    This task is suitable to detect organs on a point cloud where organs are detached from each other since
    it use the DBSCAN clustering method with a density estimator.

    Attributes
    ----------
    upstream_task : luigi.TaskParameter
        Upstream task.
        `SegmentedPointCloud` by default.
    eps : luigi.FloatParameter
        The maximum euclidean distance between two samples for one to be considered as in the neighborhood of the other.
        This is not a maximum bound on the distances of points within a cluster.
        `2.0` by default.
    min_points : luigi.IntParameter
        The number of points in a neighborhood for a point to be considered as a core point.
        This includes the point itself.
        `5` by default.

    """
    upstream_task = luigi.TaskParameter(default=SegmentedPointCloud)

    clustering_method = luigi.Parameter(default="dbscan")  # ["dbscan", "hdbscan"]
    eps = luigi.FloatParameter(default=2.0)  # dbscan
    min_points = luigi.IntParameter(default=5)  # dbscan (equivalent to 'min_sample' in hdbscan)
    min_cluster_size = luigi.IntParameter(default=5)  # hdbscan

    @staticmethod
    def get_label_pointcloud(pcd, labels, label):
        """Return a point cloud only for the selected label.

        Parameters
        ----------
        pcd : open3d.geometry.PointCloud
            A PointCloud instance with points.
        labels : list
            List of labels associated to the points.
        label : str
            Label used to select points from pointcloud.

        Returns
        open3d.geometry.PointCloud
            A point cloud containing only the points associated to the selected label.
        """
        # Get the index of points matching the semantic label
        idx_mask = np.where(np.array(labels) == label)[0]
        # Skip point cloud reconstruction if no points corresponding to label
        n_points = sum(idx_mask)
        if n_points == 0:
            print(f"No points found for label: '{label}'!")
        else:
            print(f"Found {n_points} point for, label '{label}'.")
        # Returns point cloud (colored & with normals if any):
        return pcd.select_by_index(list(idx_mask))

    def run(self):
        # Read the pointcloud from the `upstream_task`
        labelled_pcd = io.read_point_cloud(self.input_file())
        # Initialize the output FileSet object.
        output_fileset = self.output().get()
        # Get the list of semantic label ('flower', 'fruit', ...) attached to each points of the point cloud
        labels = self.input_file().get_metadata("labels")
        unique_labels = set(labels)
        # Loop on the unique set of labels:
        for label in unique_labels:
            label_pcd = self.get_label_pointcloud(labelled_pcd, labels, label)
            # Exclude stem from clustering
            if label == 'stem':
                f = output_fileset.create_file("%s_%03d" % (label, 0))
                io.write_point_cloud(f, label_pcd)
                f.set_metadata("label", label)
                continue
            # DBSCAN clustering:
            if self.clustering_method == "dbscan":
                clustered_arr = np.array(label_pcd.cluster_dbscan(eps=self.eps, min_points=self.min_points, print_progress=True))
            # HDBSCAN clustering:
            elif self.clustering_method == "hdbscan":
                pcd_arr = np.asarray(label_pcd.points)
                clusterer = HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_points, core_dist_n_jobs=-1)
                clustered_arr = np.array(clusterer.fit(pcd_arr).labels_)

            ids = np.unique(clustered_arr)
            n_ids = len(ids)
            print(f"Found {n_ids} clusters in the point cloud!")
            # For each organ
            for i in ids:
                # Exclude outliers points (-1) from output point clouds
                if i == -1:
                    continue
                cluster_pcd = self.get_label_pointcloud(label_pcd, clustered_arr, i)
                f = output_fileset.create_file("%s_%03d" % (label, i))
                io.write_point_cloud(f, cluster_pcd)
                f.set_metadata("label", label)


class CurveSkeleton(RomiTask):
    """ Creates a 3D curve skeleton.

    Module: plant3dvision.tasks.proc3d
    Default upstream tasks: TriangleMesh
    Upstream task format: ply triangle mesh
    Output task format: json with two entries "points" and "lines" (TODO: precise)

    """
    upstream_task = luigi.TaskParameter(default=TriangleMesh)

    def run(self):
        from plant3dvision import proc3d
        mesh = io.read_triangle_mesh(self.input_file())
        out = proc3d.skeletonize(mesh)
        io.write_json(self.output_file(), out)


class VoxelsWithPrior(RomiTask):
    """
    Assign class to voxel adjusting for the possibility that
    projection can be wrongly labeled.
    """
    upstream_task = luigi.TaskParameter(default=Voxels)
    recall = luigi.DictParameter(default={})
    specificity = luigi.DictParameter(default={})
    n_views = luigi.IntParameter()

    def run(self):
        prediction_file = self.upstream_task().output().get().get_files()[0]
        voxels = io.read_npz(prediction_file)
        out = {}
        l = list(voxels.keys())
        for c in l:
            if c in self.recall:
                recall = self.recall[c]
            else:
                continue
            if c in self.specificity:
                specificity = self.specificity[c]
            else:
                continue
            l0 = (self.n_views - voxels[c]) * np.log(specificity) + voxels[
                c] * np.log(1 - specificity)
            l1 = (self.n_views - voxels[c]) * np.log(1 - recall) + voxels[
                c] * np.log(recall)
            out[c] = l1 - l0

        outfs = self.output().get()
        outfile = self.output_file()
        io.write_npz(outfile, out)
        outfile.set_metadata(prediction_file.get_metadata())
