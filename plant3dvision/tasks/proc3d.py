#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import luigi
import numpy as np
import open3d as o3d

from plant3dvision import proc3d
from plant3dvision.tasks import config
from plant3dvision.tasks.cl import Voxels
from plant3dvision.tasks.colmap import Colmap
from plant3dvision.tasks.proc2d import Segmentation2D
from plantdb import io
from romitask import RomiTask
from romitask.log import configure_logger

logger = configure_logger(__name__)


class PointCloud(RomiTask):
    """Computes a point cloud from volumetric voxel data (either single or multiclass).

    Attributes
    ----------
    upstream_task : luigi.TaskParameter, optional
        Upstream task that generate the volume.
        Restricted to ``'Voxels'`` for now.
        Defaults to ``'Voxels'``.
    scan_id : luigi.Parameter, optional
        The dataset id (scan name) to use to create the ``FilesetTarget``.
        If unspecified (default), the current active scan will be used.
    level_set_value : luigi.FloatParameter, optional
        ???
        Defaults to ``1.0``.
    background_prior : luigi.FloatParameter, optional
        ???
        Used only if `labels` were defined in upstream tasks (multiclass).
        Defaults to ``1.0``.
    min_contrast : luigi.FloatParameter, optional
        ???
        Used only if `labels` were defined in upstream tasks (multiclass).
        Defaults to ``10.0``.
    min_score : luigi.FloatParameter, optional
        ???
        Used only if `labels` were defined in upstream tasks (multiclass).
        Defaults to ``0.2``.

    See Also
    --------
    plant3dvision.proc3d.vol2pcd

    Notes
    -----
    Task output is a single PLY file with the point cloud.

    Metadata may include label name if multiclass.
    """
    upstream_task = luigi.TaskParameter(default=Voxels)  # override default attribute from ``RomiTask``
    level_set_value = luigi.FloatParameter(default=1.0)

    background_prior = luigi.FloatParameter(default=1.0)  # only used if labels were defined (multiclass)
    min_contrast = luigi.FloatParameter(default=10.0)  # only used if labels were defined (multiclass)
    min_score = luigi.FloatParameter(default=0.2)  # only used if labels were defined (multiclass)

    def run(self):
        ifile = self.input_file()
        # Guess if it's a labelled volume:
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

                    out = proc3d.vol2pcd(pred_c, origin, voxel_size, self.level_set_value)
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
            out = proc3d.vol2pcd(voxels, origin, voxel_size, self.level_set_value)
            io.write_point_cloud(self.output_file(), out)
            self.output_file().set_metadata({'voxel_size': voxel_size})


class SegmentedPointCloud(RomiTask):
    """Segments an existing point cloud using 2D pictures.

    Attributes
    ----------
    upstream_task : luigi.TaskParameter, optional
        Task upstream of this task, should provide a point cloud.
        Should be either ``Colmap`` or ``PointCloud``.
        Defaults to ``Colmap``.
    scan_id : luigi.Parameter, optional
        The dataset id (scan name) to use to create the ``FilesetTarget``.
        If unspecified (default), the current active scan will be used.
    upstream_segmentation : luigi.TaskParameter, optional
        Task upstream of this task, should provide a 2D segmentation of the 'images'.
        Defaults to ``Segmentation2D``.
    use_colmap_poses : luigi.BoolParameter, optional
        Defaults to ``True``.

    See Also
    --------
    plant3dvision.proc3d.backproject_points

    Notes
    -----
    Task output is a single PLY file with the colored (labelled) point cloud.

    If the upstream task is set to ``Colmap``, the dense point cloud have to be reconstructed by COLMAP.
    """
    upstream_task = luigi.TaskParameter(default=Colmap)  # override default attribute from ``RomiTask``
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
            pixels = np.asarray(proc3d.backproject_points(pts, K, rotmat, tvec) + 0.5, dtype=int)

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
    """Triangulates input point cloud.

    Attributes
    ----------
    upstream_task  : luigi.TaskParameter, optional
        Task upstream of this task, should provide a point cloud.
        Defaults to ``PointCloud``.
    scan_id : luigi.Parameter, optional
        The dataset id (scan name) to use to create the ``FilesetTarget``.
        If unspecified (default), the current active scan will be used.
    library : luigi.Parameter, optional
        The library to mesh the point cloud, choose either "cgal" or "open3d".
        If "cgal", use the ``poisson_mesh`` method from the CGAL library, see [CGAL]_.
        If "open3d", use the ``create_from_point_cloud_poisson`` method from the Open3D library, see [Open3D]_.
        Defaults to ``"open3d"``.
    filtering : luigi.Parameter, optional
        The filtering method to apply to obtained triangle mesh, if any.
        Valid choices are "most connected triangles", "largest connected triangles" or "".
        "most connected triangles": get the largest cluster of triangles in terms of "number of triangles".
        "largest connected triangles": get the largest cluster of triangles in terms of "total triangle area".
        Defaults to ``"most connected triangles"``, use ``""`` to deactivate filtering.
    depth : luigi.IntParameter, optional
        Depth parameter used by Open3D to mesh the point cloud, see [o3d_tri_poisson]_ for more details.
        Defaults to ``9``.

    See Also
    --------
    plant3dvision.proc3d.pcd2mesh
    cgal.poisson_mesh
    o3d.geometry.TriangleMesh.create_from_point_cloud_poisson

    Notes
    -----
    Currently, ignores class data and needs only one connected component, use ``ClusteredMesh`` instead.

    Task output is a single PLY file with the triangular mesh.

    References
    ----------
    .. [CGAL] `Poisson Surface Reconstruction user manual <https://doc.cgal.org/latest/Poisson_surface_reconstruction_3/index.html>`_.
    .. [o3d_tri_poisson] `Open3D's TriangleMesh API <http://www.open3d.org/docs/latest/python_api/open3d.geometry.TriangleMesh.html#open3d.geometry.TriangleMesh.create_from_point_cloud_poisson>`_.
    """
    upstream_task = luigi.TaskParameter(default=PointCloud)  # override default attribute from ``RomiTask``
    library = luigi.Parameter(default="open3d")  # ["cgal", "open3d"]
    filtering = luigi.Parameter(
        default="most connected triangles")  # ["", "most connected triangles", "largest connected triangles", "dbscan point cloud"]

    depth = luigi.IntParameter(default=9)  # used by open3d library

    def run(self):
        from plant3dvision import proc3d
        point_cloud = io.read_point_cloud(self.input_file())

        # TODO: Add DBSCAN clustering method to filter the point cloud prior to meshing
        if self.filtering == "dbscan point cloud":
            raise NotImplementedError("Coming soon!")

        if self.library == "cgal":
            out = proc3d.pcd2mesh(point_cloud)
        elif self.library == "open3d":
            out, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=self.depth)

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
    """Triangulate input labelled point cloud.

    Attributes
    ----------
    upstream_task : luigi.TaskParameter, optional
        The task upstream to this one, should provide a segmented point cloud.
        Defaults to ``SegmentedPointCloud``.
    scan_id : luigi.Parameter, optional
        The dataset id (scan name) to use to create the ``FilesetTarget``.
        If unspecified (default), the current active scan will be used.
    depth : luigi.IntParameter, optional
        Depth parameter used by Open3D to mesh the point cloud, see [o3d_tri_poisson]_ for more details.
        Defaults to ``9``.

    See Also
    --------
    o3d.geometry.TriangleMesh.create_from_point_cloud_poisson

    Notes
    -----
    Task outputs are a series of PLY file with a triangular mesh for each label.

    """
    upstream_task = luigi.TaskParameter(default=SegmentedPointCloud)  # override default attribute from ``RomiTask``

    depth = luigi.IntParameter(default=9)  # used by open3d library

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
            t, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=self.depth)
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

    Attributes
    ----------
    upstream_task : luigi.TaskParameter, optional
        The task upstream to this one, should provide a segmented point cloud.
        Defaults to ``SegmentedPointCloud``.
    scan_id : luigi.Parameter, optional
        The dataset id (scan name) to use to create the ``FilesetTarget``.
        If unspecified (default), the current active scan will be used.
    eps : luigi.FloatParameter, optional
        The maximum Euclidean distance between two samples for one to be considered as in the neighborhood of the other.
        This is not a maximum bound on the distances of points within a cluster.
        Defaults to ``2.0``.
    min_points : luigi.IntParameter, optional
        The number of points in a neighborhood for a point to be considered as a core point.
        This includes the point itself.
        Defaults to ``5``.

    See Also
    --------
    open3d.geometry.PointCloud.cluster_dbscan

    Notes
    -----
    This is done for each semantic label ('flower', 'fruit', ...) of the labelled point cloud,
    except for the stem as it is considered to be one organ.
    This task is suitable to detect organs on a point cloud where organs are detached from each other since
    it use the DBSCAN clustering method with a density estimator.
    See [o3d_pcd_cluster_dbscan] & [DBSCAN]_ for more details.

    References
    ----------
    .. [o3d_pcd_cluster_dbscan] `Open3D's PointCloud API <http://www.open3d.org/docs/latest/python_api/open3d.geometry.PointCloud.html#open3d.geometry.PointCloud.cluster_dbscan>`_.
    .. [DBSCAN] `Scikit-learn user guide for DBSCAN <https://scikit-learn.org/stable/modules/clustering.html#dbscan>`_.

    """
    upstream_task = luigi.TaskParameter(default=SegmentedPointCloud)  # override default attribute from ``RomiTask``
    eps = luigi.FloatParameter(default=2.0)
    min_points = luigi.IntParameter(default=5)

    def get_label_pointcloud(self, pcd, labels, label):
        """Return a point cloud only for the selected label.

        Parameters
        ----------
        pcd : open3d.geometry.PointCloud
            A PointCloud instance with points.
        labels : list
            The list of labels associated to the points.
        label : str
            Label used to select points from point cloud.

        Returns
        -------
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
        # Read the point cloud from the `upstream_task`
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
            clustered_arr = np.array(
                label_pcd.cluster_dbscan(eps=self.eps, min_points=self.min_points, print_progress=True))

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
    """Creates a 3D curve skeleton from a triangular mesh.

    Attributes
    ----------
    upstream_task : luigi.TaskParameter
        The task upstream to this one, should provide a triangular mesh.
        Defaults to ``TriangleMesh``.
    scan_id : luigi.Parameter, optional
        The dataset id (scan name) to use to create the ``FilesetTarget``.
        If unspecified (default), the current active scan will be used.

    See Also
    --------
    plant3dvision.proc3d.skeletonize

    Notes
    -----
    Task output is a JSON file with two entries, "points" and "lines".
    """
    upstream_task = luigi.TaskParameter(default=TriangleMesh)  # override default attribute from ``RomiTask``

    def run(self):
        task_name = self.get_task_family()
        uptask_name = self.upstream_task.get_task_family()

        if uptask_name == "TriangleMesh":
            from plant3dvision import proc3d
            mesh = io.read_triangle_mesh(self.input_file())
            out = proc3d.skeletonize(mesh)
        else:
            logger.error(f"No implementation to compute `{task_name}` from `{uptask_name}`.")
            logger.info(f"Select `upstream_task` among: 'TriangleMesh'.")
            raise NotImplementedError(f"No implementation to compute `{task_name}` from `{task_name}`.")
        io.write_json(self.output_file(), out)


class RefineSkeleton(RomiTask):
    """Refine a 3D curve skeleton using stochastic deformation registration.


    Attributes
    ----------
    upstream_task : luigi.TaskParameter
        The task upstream to this one, should provide a triangular mesh.
        Defaults to ``CurveSkeleton``.
    upstream_task : luigi.TaskParameter
        The task providing the point cloud to refine the skeleton from.
        Defaults to ``PointCloud``.
    scan_id : luigi.Parameter, optional
        The dataset id (scan name) to use to create the ``FilesetTarget``.
        If unspecified (default), the current active scan will be used.

    See Also
    --------
    skeleton_refinement.stochastic_registration.perform_registration

    Notes
    -----
    Task output is a JSON file with two entries, "points" and "lines".
    """
    upstream_task = luigi.TaskParameter(default=CurveSkeleton)  # override default attribute from ``RomiTask``
    upstream_pcd = luigi.TaskParameter(default=PointCloud)
    alpha = luigi.FloatParameter(default=2.)
    beta = luigi.FloatParameter(default=2.)
    max_iterations = luigi.IntParameter(default=100)
    tolerance = luigi.FloatParameter(default=0.001)

    def requires(self):
        return {"skeleton": self.upstream_task(), "pcd": self.upstream_pcd()}

    def run(self):
        from skeleton_refinement.stochastic_registration import perform_registration
        skel = io.read_json(self.input()["skeleton"].get().get_files()[0])
        pcd = io.read_point_cloud(self.input()["pcd"].get().get_files()[0])
        refined_skel = perform_registration(np.asarray(pcd.points), np.array(skel["points"]),
                                            alpha=self.alpha, beta=self.beta, max_iterations=self.max_iterations)
        refined_skel = {"points": refined_skel.tolist(), "lines": skel['lines']}
        io.write_json(self.output_file(), refined_skel)


class VoxelsWithPrior(RomiTask):
    """Assign class to voxel adjusting for the possibility that projection can be wrongly labeled.

    Attributes
    ----------
    upstream_task : luigi.TaskParameter, optional
        The task upstream to this one, should provide an NPZ voxel volume.
        Defaults to ``Voxels``.
    scan_id : luigi.Parameter, optional
        The dataset id (scan name) to use to create the ``FilesetTarget``.
        If unspecified (default), the current active scan will be used.
    recall : luigi.DictParameter, optional
        ???.
        Defaults to ``{}``.
    specificity : luigi.DictParameter, optional
        ???.
        Defaults to ``{}``.
    n_views : luigi.IntParameter
        ???.

    Notes
    -----
    Upstream task format: NPZ (multiclass) voxel volume file.
    Output task format: NPZ (multiclass) voxel volume file.

    """
    upstream_task = luigi.TaskParameter(default=Voxels)
    recall = luigi.DictParameter(default={})
    specificity = luigi.DictParameter(default={})
    n_views = luigi.IntParameter()

    def run(self):
        prediction_file = self.upstream_task().output().get().get_files()[0]
        voxels = io.read_npz(prediction_file)
        out = {}
        labels = list(voxels.keys())
        for label in labels:
            if label in self.recall:
                recall = self.recall[label]
            else:
                continue
            if label in self.specificity:
                specificity = self.specificity[label]
            else:
                continue
            l0 = (self.n_views - voxels[label]) * np.log(specificity) + voxels[label] * np.log(1 - specificity)
            l1 = (self.n_views - voxels[label]) * np.log(1 - recall) + voxels[label] * np.log(recall)
            out[label] = l1 - l0

        outfs = self.output().get()
        outfile = self.output_file()
        io.write_npz(outfile, out)
        outfile.set_metadata(prediction_file.get_metadata())
