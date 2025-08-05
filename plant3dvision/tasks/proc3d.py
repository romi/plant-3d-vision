#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import luigi
import numpy as np
import open3d as o3d
from plantdb.commons import io

from plant3dvision import proc3d
from plant3dvision.tasks import config
from plant3dvision.tasks.voxel_reconstruction import Voxels
from plant3dvision.tasks.colmap import Colmap
from plant3dvision.tasks.proc2d import Segmentation2D
from romitask import RomiTask
from romitask.log import get_logger
from skeleton_refinement.stochastic_registration import knn_mst

logger = get_logger(__name__)


class PointCloud(RomiTask):
    """Generate a 3D point cloud from volumetric/voxel data.

    This task processes either a single-class or multi-class volume by extracting
    features based on a specified level set value and optional thresholds (contrast
    and score). For multi-class volumes, data are filtered and combined by applying
    a background prior and thresholding rules, and label information is attached to
    the output point cloud. The final result is stored as a PLY file with associated
    metadata such as the data origin, voxel size, and (if applicable) label names.

    Parameters
    ----------
    upstream_task : luigi.TaskParameter, optional
        The upstream task providing the input data for this task. Defaults to ``Voxels``.
    level_set_value : luigi.FloatParameter, optional
        Value used to define the level set for point cloud generation. Default is ``1.0``.
    labels : luigi.ListParameters, optional
        List of class labels to process. An empty list (default) processes a single unlabeled volume.
        A single label processes that specific class. Multiple labels trigger multi-class processing.
        Defaults to ``[]``.
    background_prior : luigi.FloatParameter, optional
        Prior weight applied to the background class when processing multi-class volumes.
        Defaults to ``1.0``.
    min_contrast : luigi.FloatParameter, optional
        Minimum contrast ratio to consider for class predictions in a multi-class volume.
        Defaults to ``10.0``.
    min_score : luigi.FloatParameter, optional
        Minimum score threshold for class predictions in a multi-class volume.
        Defaults to ``0.2``.

    Returns
    -------
    romitask.task.FilesetTarget
        A PLY file containing the (labelled) point cloud.

    See Also
    --------
    plant3dvision.proc3d.vol2pcd : Core function used for volume to point cloud conversion

    Notes
    -----
    For multi-class volumes:
    - Classes are processed based on highest probability per voxel
    - Background class is weighted by the background_prior
    - Points are filtered based on contrast between the highest and second-highest class
    - Points are filtered based on minimum score threshold
    - Each class gets a color from the configuration or a random color

    For single-class volumes:
    - A point cloud is generated directly from the volumetric data
    - The level_set_value determines the isosurface extraction

    The output is a PLY file containing the point cloud with associated metadata.
    If multi-class, point label information is included in the metadata.
    """
    upstream_task = luigi.TaskParameter(default=Voxels)  # override default attribute from ``RomiTask``
    level_set_value = luigi.FloatParameter(default=1.0)

    labels = luigi.ListParameter(default=[])
    background_prior = luigi.FloatParameter(default=1.0)  # only used if labels were defined (multiclass)
    min_contrast = luigi.FloatParameter(default=10.0)  # only used if labels were defined (multiclass)
    min_score = luigi.FloatParameter(default=0.2)  # only used if labels were defined (multiclass)

    def run_multiclass(self, labels):
        """Processes multi-class voxel data to generate a unified point cloud.

        Parameters
        ----------
        labels : list of str
            A list of class labels to process. Each label corresponds to a
            voxel class in the input volume data.

        Raises
        ------
        FileNotFoundError
            If the input files corresponding to any of the specified labels
            do not exist.
        ValueError
            If voxel data or metadata is improperly formatted or missing.
        TypeError
            If labels or voxel data contain invalid types.

        Notes
        -----
        - This method performs multi-class processing by iterating over the
          provided labels and aggregating voxel data into a multi-dimensional
          array.
        - The algorithm applies class-specific modifications, such as adjusting
          the background voxel values using a prior and filtering voxels based
          on contrast and score thresholds.
        - A point cloud is generated for each class based on its filtered voxel
          data. Each point cloud shares metadata on origin and voxel size.
        - Points are colorized and added to a final aggregated point cloud.
          Predefined colors are used when available; otherwise, random colors
          are assigned.
        - Finally, the point cloud and labels for all points are saved as
          outputs.
        """
        for label in labels:
            ifile = self.input_file(suffix=label)
            voxels = io.read_volume(ifile)
            # Collect the names of all classes
            label = list(voxels.keys())
            # Prepare an array to aggregate voxel data from all classes
            res = np.zeros((*voxels[label[0]].shape, len(label)))
            # Stack each class in a new dimension
            for i in range(len(label)):
                res[:, :, :, i] = voxels[label[i]]
            # Apply background prior if class is 'background'
            for i in range(len(label)):
                if label[i] == 'background':
                    res[:, :, :, i] *= self.background_prior
            # Determine the index of the class with the highest value per voxel
            res_idx = np.argmax(res, axis=3)
            # Prepare an Open3D point cloud object for aggregation
            pcd = o3d.geometry.PointCloud()
            # Fetch metadata for origin and voxel size
            origin = np.array(ifile.get_metadata('origin'))
            voxel_size = float(ifile.get_metadata('voxel_size'))
            # List to keep track of assigned labels for each point
            point_labels = []
            # Predefined color dictionary for known labels
            colors = config.PointCloudColorConfig().colors
            # Iterate over all labels to generate point clouds
            for i in range(len(label)):
                logger.debug(f"label = {label[i]}")
                # Skip background in point cloud generation
                if label[i] == 'background':
                    continue
                # Compute the maximum values across all other classes
                pred_no_c = np.max(np.delete(res, i, axis=3), axis=3)
                # Identify voxels belonging to the current class
                pred_c = (res_idx == i)
                # Apply contrast threshold if min_contrast > 1.0
                if self.min_contrast > 1.0:
                    pred_c *= (pred_c > (self.min_contrast * pred_no_c))
                # Apply minimum score threshold
                pred_c *= (pred_c > self.min_score)
                # Convert filtered volume to a partial point cloud
                out = proc3d.vol2pcd(pred_c, origin, voxel_size, self.level_set_value)
                # Assign a color to all points in this partial cloud
                color = np.zeros((len(out.points), 3))
                if label[i] in colors:
                    color[:] = np.asarray(colors[label[i]])
                else:
                    # Generate a random color if none is predefined
                    color[:] = np.random.rand(3)
                # Set point colors and add to the final point cloud
                color = o3d.utility.Vector3dVector(color)
                out.colors = color
                pcd = pcd + out
                # Collect label info for this subset of points
                point_labels = point_labels + [label[i]] * len(out.points)
        # Save the combined point cloud and label information
        io.write_point_cloud(self.output_file(create=True), pcd)
        self.output_file(create=True).set_metadata({'labels': point_labels})

    def run_single_class(self, ifile):
        """Processes a binary volume to generate a point cloud.

        This method handles a single-class volumetric dataset by converting it into
        a point cloud, derived from the specified isosurface level set value. The
        resulting point cloud is saved to a file in addition to attaching metadata
        such as voxel size. Metadata like 'origin' and 'voxel_size' are extracted
        directly from the input file.

        Parameters
        ----------
        ifile : File
            The input file containing volumetric data and related metadata.
        """
        voxels = io.read_volume(ifile)
        # For single-class data, fetch origin and voxel size as usual
        origin = np.array(ifile.get_metadata('origin'))
        voxel_size = float(ifile.get_metadata('voxel_size'))
        # Directly create a point cloud from the single volume
        out = proc3d.vol2pcd(voxels, origin, voxel_size, self.level_set_value)
        # Write the point cloud to file and attach metadata
        io.write_point_cloud(self.output_file(create=True), out)
        self.output_file(create=True).set_metadata({'voxel_size': voxel_size})

    def run(self):
        """Process a volumetric data file into a point cloud representation.

        Notes
        ----
        The resulting point cloud is written to an output file with metadata.

        If the input is multi-class, the method combines voxel data from multiple classes while applying
        optional thresholds like contrast and score. Single-class data is directly processed for point
        cloud generation.
        """
        # Case 1: No labels specified - process default single class volume
        if len(self.labels) == 0:
            ifile = self.input_file()
            self.run_single_class(ifile)

        # Case 2: One label specified - process single class with label suffix
        elif len(self.labels) == 1:
            # Get input file with label name appended to filename
            ifile = self.input_file(suffix=f"_{self.labels[0]}")
            self.run_single_class(ifile)

        # Case 3: Multiple labels - combine data from multiple classes
        else:
            # Process multiple class volumes and combine them into single point cloud
            self.run_multiclass(self.labels)


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
        """Loads a point cloud from a file generated by the upstream task.

        Returns
        -------
        open3d.geometry.PointCloud
            The loaded point cloud object containing the 3D data points.

        Raises
        ------
        FileNotFoundError
            If the specified file or any files are not found during retrieval.
        """
        try:
            x = self.upstream_task().output().get().get_file("dense")
        except FileNotFoundError:
            x = self.upstream_task().output().get().get_files()[0]

        return io.read_point_cloud(x)

    def is_in_pict(self, px, shape):
        """Checks whether a given point (pixel) is within the bounds of an image with a given shape.

        This function evaluates whether a specified pixel coordinate lies inside
        the valid boundaries of an image defined by its shape (rows and columns).
        It ensures the pixel's horizontal and vertical coordinates are within the
        permissible range.

        Parameters
        ----------
        px : tuple of int
            A tuple specifying the (x, y) coordinates of the pixel being checked.
        shape : tuple of int
            A tuple defining the shape of the image in terms of
            (number of rows, number of columns).

        Returns
        -------
        bool
            True if the pixel is within the bounds of the image, False otherwise.
        """
        return px[0] >= 0 and px[0] < shape[1] and px[1] >= 0 and px[1] < shape[0]

    def run(self):
        """Processes segmented images to assign point labels and colors to a point cloud based on camera poses and image data.

        This function processes camera pose metadata for accurate back-projection of the 3D points to the image space,
        reads label data from the input files, and computes the label with the highest score for each point.
        Additionally, the function updates the point cloud's color representation.

        Main steps:
        - Loads point cloud data and retrieves segmentation files.
        - Computes scores for associating point labels to each 3D point.
        - Supports camera poses from COLMAP or an alternate source for back-projection.
        - Assigns colors to points based on associated labels.
        - Outputs labeled and colorized point cloud with metadata.

        Raises
        ------
        Exceptions may be raised by underlying operations, such as file access, data processing, or library calls, if inputs
        are invalid or required metadata is incomplete.
        """
        # Get segmentation files from upstream task
        fs = self.upstream_segmentation().output().get()
        # Load and convert point cloud to numpy array
        pcd = self.load_point_cloud()
        pts = np.asarray(pcd.points)

        # Extract unique labels from segmentation files
        labels = set()
        for fi in fs.get_files():
            label = fi.get_metadata('channel')
            if label is not None:
                labels.add(label)
        labels = list(labels)
        # Remove special labels that should not be processed
        labels.remove('background')
        if 'rgb' in labels:
            labels.remove('rgb')

        # Initialize scores matrix (labels × points)
        scores = np.zeros((len(labels), len(pts)))

        # Process each segmentation file
        for fi in fs.get_files():
            label = fi.get_metadata("channel")
            if label not in labels:
                continue

            # Get camera parameters based on source (COLMAP or alternative)
            if self.use_colmap_poses:
                camera = fi.get_metadata("colmap_camera")
            else:
                camera = fi.get_metadata("camera")
            if camera is None:
                logger.warning(f"Could not get camera pose for view, skipping...")
                continue

            # Extract camera parameters for back-projection
            rotmat = np.array(camera["rotmat"])
            tvec = np.array(camera["tvec"])
            intrinsics = camera["camera_model"]["params"]
            # Construct camera matrix K from intrinsics
            K = np.array([[intrinsics[0], 0, intrinsics[2]],
                          [0, intrinsics[1], intrinsics[3]],
                          [0, 0, 1]])

            # Back-project 3D points to 2D image coordinates
            pixels = np.asarray(proc3d.backproject_points(pts, K, rotmat, tvec) + 0.5, dtype=int)
            label_idx = labels.index(label)
            mask = io.read_image(fi)

            # Accumulate scores for each point based on mask values
            for i, px in enumerate(pixels):
                if self.is_in_pict(px, mask.shape):
                    scores[label_idx, i] += mask[px[1], px[0]]

        # Determine final label for each point based on highest score
        pts_labels = np.argmax(scores, axis=0).flatten()
        logger.critical(f"Processed following labels: {labels}")

        # Get color mapping from config
        colors = config.PointCloudColorConfig().colors
        logger.critical(f"Associated colors: {colors}")

        # Initialize arrays for point colors and labels
        color_array = np.zeros((len(pts), 3))
        point_labels = [""] * len(pts)

        # Assign colors and labels to points
        for i in range(len(labels)):
            nlab_pts = (pts_labels == i).sum()
            logger.critical(f"Number of points associated to label '{labels[i]}': {nlab_pts}")

            # Use predefined color if available, otherwise random color
            if labels[i] in colors:
                color_array[pts_labels == i, :] = np.asarray(colors[labels[i]])
            else:
                color_array[pts_labels == i, :] = np.random.rand(3)

            # Store label names for each point
            l = np.nonzero(pts_labels == i)[0].tolist()
            for u in l:
                point_labels[u] = labels[i]

        # Update point cloud colors and save results
        pcd.colors = o3d.utility.Vector3dVector(color_array)
        out = self.output_file(create=True)
        io.write_point_cloud(out, pcd)
        out.set_metadata("labels", point_labels)


class TriangleMesh(RomiTask):
    """Triangulates a 3D point cloud to create a triangle mesh.

    This task creates a triangular mesh from an input point cloud using
    either CGAL or Open3D libraries. The mesh can be filtered to retain only
    the largest connected component based on different criteria.

    Parameters
    ----------
    upstream_task  : luigi.TaskParameter, optional
        Task upstream of this task, should provide a point cloud.
        Defaults to ``PointCloud``.
    scan_id : luigi.Parameter, optional
        The dataset id (scan name) to use to create the ``FilesetTarget``.
        If unspecified (default), the current active scan will be used.
    library : luigi.Parameter, optional
        The library to mesh the point cloud. Options:
        - "cgal": use the ``poisson_mesh`` method from the CGAL library
        - "open3d": use the ``create_from_point_cloud_poisson`` method from Open3D
        Default is "open3d".
    filtering : luigi.Parameter, optional
        The filtering method to apply to the triangle mesh. Options:
        - "most connected triangles": get the largest cluster by number of triangles
        - "largest connected triangles": get the largest cluster by total triangle area
        - "": no filtering
        Default is "most connected triangles".
    depth : luigi.IntParameter, optional
        Depth parameter used by Open3D to mesh the point cloud.
        Controls the resolution of the resulting mesh.
        Higher values create finer meshes but require more computation, see [o3d_tri_poisson]_ for more details.
        Defaults to ``9``.

    Returns
    -------
    romitask.task.FilesetTarget
        A PLY file containing the triangular mesh.

    See Also
    --------
    plant3dvision.proc3d.pcd2mesh
    cgal.poisson_mesh
    o3d.geometry.TriangleMesh.create_from_point_cloud_poisson

    Notes
    -----
    Currently ignores class data and needs only one connected component.
    For more sophisticated mesh clustering, use ``ClusteredMesh`` instead.

    The task output is a single PLY file with the triangular mesh.

    When using "open3d" library, the depth parameter significantly affects
    the mesh quality and processing time. Higher values (9-11) produce finer
    meshes but take longer to compute.

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

        io.write_triangle_mesh(self.output_file(create=True), out)


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
        # Read input point cloud file
        x = io.read_point_cloud(self.input_file())
        # Convert point cloud data to numpy arrays for efficient processing
        all_points = np.asarray(x.points)
        all_normals = np.asarray(x.normals)
        all_colors = np.asarray(x.colors)

        # Get semantic labels for each point (e.g., 'flower', 'fruit')
        labels = self.input_file().get_metadata("labels")
        output_fileset = self.output().get()

        # Process each unique semantic label separately
        for l in set(labels):
            pcd = o3d.geometry.PointCloud()
            # Find indices of points with current label
            idx = [i for i in range(len(labels)) if labels[i] == l]
            # Extract points, normals, and colors for current label
            points = all_points[idx, :]
            normals = all_normals[idx, :]
            colors = all_colors[idx, :]

            # Skip if no points found for current label
            if len(points) == 0:
                logger.critical(f"No points found for label: '{l}'")
                continue

            # Create point cloud for current label
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            # Generate mesh using Poisson surface reconstruction
            t, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=self.depth)
            # Identify connected components in the mesh
            t.compute_adjacency_list()
            k, cc, _ = t.cluster_connected_triangles()
            k = np.asarray(k)
            tri_np = np.asarray(t.triangles)

            # Create separate mesh for each connected component
            for j in range(len(cc)):
                # Extract triangles for current component
                newt = o3d.geometry.TriangleMesh(t.vertices,
                                                 o3d.utility.Vector3iVector(tri_np[k == j, :]))
                newt.vertex_colors = t.vertex_colors
                newt.remove_unreferenced_vertices()
                # Save mesh component to file with label metadata
                f = output_fileset.create_file(f"{l}_{j:03d}")
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
        # Load point cloud data from input file
        labelled_pcd = io.read_point_cloud(self.input_file())
        # Get output FileSet for storing results
        output_fileset = self.output().get()

        # Get semantic labels for each point in the cloud
        labels = self.input_file().get_metadata("labels")
        unique_labels = set(labels)  # Get unique organ labels (flower, fruit, etc.)

        # Process each unique organ label separately
        for label in unique_labels:
            # Extract points corresponding to current label
            label_pcd = self.get_label_pointcloud(labelled_pcd, labels, label)
            # Special handling for stem - no clustering needed
            if label == 'stem':
                f = output_fileset.create_file(f"{label}_000")
                io.write_point_cloud(f, label_pcd)
                f.set_metadata("label", label)
                continue
            # Perform DBSCAN clustering on non-stem organs
            clustered_arr = np.array(
                label_pcd.cluster_dbscan(
                    eps=self.eps,  # Max distance between points in a cluster
                    min_points=self.min_points,  # Min points to form a cluster
                    print_progress=True
                )
            )
            # Get unique cluster IDs (-1 represents noise points)
            ids = np.unique(clustered_arr)
            n_ids = len(ids)
            print(f"Found {n_ids} clusters in the point cloud!")
            # Process each cluster separately
            for i in ids:
                # Skip noise points (cluster ID -1)
                if i == -1:
                    continue
                # Extract points for current cluster
                cluster_pcd = self.get_label_pointcloud(label_pcd, clustered_arr, i)
                # Save cluster point cloud to output file
                f = output_fileset.create_file(f"{label}_{i:03d}")
                io.write_point_cloud(f, cluster_pcd)
                f.set_metadata("label", label)


class CurveSkeleton(RomiTask):
    """Creates a 3D curve skeleton from a triangular mesh.

    This class implements a task that generates a curve skeleton representation from a
    triangular mesh. The skeleton consists of points and lines that capture the essential
    topological structure of the 3D shape.

    Parameters
    ----------
    upstream_task : luigi.TaskParameter
        The task upstream to this one, should provide a triangular mesh.
        Defaults to ``TriangleMesh``.
    scan_id : luigi.Parameter, optional
        The dataset id (scan name) to use to create the ``FilesetTarget``.
        If unspecified (default), the current active scan will be used.

    Returns
    -------
    romitask.task.FilesetTarget
        A JSON file containing the skeleton data with two keys:
        - "points": The 3D coordinates of the skeleton vertices
        - "lines": The connectivity information defining the skeleton edges

    Raises
    ------
    NotImplementedError
        If the upstream task is not supported (currently only supports ``TriangleMesh``).

    See Also
    --------
    plant3dvision.proc3d.skeletonize : Core function used to generate the skeleton.

    Notes
    -----
    Task output is a JSON file with two entries, "points" and "lines".
    "points" contains the 3D coordinates of vertices in the skeleton.
    "lines" contains pairs of vertex indices that define the connections.

    The skeletonization algorithm is implemented in the `plant3dvision.proc3d` module.
    Only triangular meshes from the ``TriangleMesh`` task are currently supported as input.
    """
    upstream_task = luigi.TaskParameter(default=TriangleMesh)  # override default attribute from ``RomiTask``

    def run(self):
        # Get the task names for current and upstream tasks
        task_name = self.get_task_family()
        uptask_name = self.upstream_task.get_task_family()

        # Check if upstream task is a TriangleMesh task
        if uptask_name == "TriangleMesh":
            from plant3dvision import proc3d
            # Read the triangular mesh from input file
            mesh = io.read_triangle_mesh(self.input_file())
            # Generate curve skeleton from mesh
            out = proc3d.skeletonize(mesh)
        else:
            # Raise error if upstream task is not supported
            logger.error(f"No implementation to compute `{task_name}` from `{uptask_name}`.")
            logger.info(f"Select `upstream_task` among: 'TriangleMesh'.")
            raise NotImplementedError(f"No implementation to compute `{task_name}` from `{task_name}`.")

        # Save the skeleton data (points and lines) as JSON file
        io.write_json(self.output_file(create=True), out)


class RefineSkeleton(RomiTask):
    """Refine a 3D curve skeleton using stochastic deformation registration.

    This class implements a ROMI task that refines an existing 3D curve skeleton by
    using stochastic deformation registration against a point cloud.
    The refinement process adjusts skeleton vertices to better match the underlying point cloud data
    while maintaining the overall structure.
    An optional step can reconstruct the skeleton connectivity using a minimum
    spanning tree on a k-nearest neighbor graph.

    Parameters
    ----------
    upstream_task : luigi.TaskParameter
        The task upstream to this one, should provide a triangular mesh.
        Defaults to ``CurveSkeleton``.
    upstream_pcd : luigi.TaskParameter
        The task providing the point cloud to refine the skeleton from.
        Defaults to ``PointCloud``.
    scan_id : luigi.Parameter, optional
        The dataset id (scan name) to use to create the ``FilesetTarget``.
        If unspecified (default), the current active scan will be used.
    alpha : luigi.FloatParameter, optional
        The alpha value controlling the stiffness term in skeleton refinement.
        Higher values result in less deformation.
        Defaults to `5.`.
    beta : luigi.FloatParameter, optional
        The beta value controlling the regularization strength in skeleton refinement.
        Higher values result in smoother deformation.
        Defaults to `5.`.
    max_iterations : luigi.IntParameter, optional
        Maximum number of iterations of the EM algorithm to perform.
        Defaults to `100`.
    tolerance : luigi.FloatParameter, optional
        Convergence tolerance to use to stop the iterations of the EM algorithm.
        Defaults to `0.0001`.
    knn_mst : luigi.BoolParameter, optional
        Whether to perform an update of the skeleton using the minimum spanning tree on knn-graph.
        If ``False``, original connectivity is kept. Defaults to `True`.
    n_neighbors : luigi.IntParameter, optional
        The number of neighbors to search for in the skeleton points when creating the knn-graph.
        Only used if `knn_mst` is ``True``. Defaults to `5`.
    knn_algorithm : luigi.Parameter, optional
        The algorithm to use for computing the kNN distance.
        Defaults to `kd_tree`, valid choices are 'auto', 'ball_tree', 'kd_tree' or 'brute'.
    mst_algorithm : luigi.Parameter, optional
        The algorithm to use for computing the minimum spanning tree.
        Defaults to `kruskal`, valid choices are 'kruskal', 'prim' or 'boruvka'.

    Returns
    -------
    romitask.task.FilesetTarget
        A FilesetTarget containing a JSON file with the refined skeleton data.
        The JSON has two keys:
        - "points": list of lists, 3D coordinates of skeleton points
        - "lines": list of tuples, connectivity information as pairs of point indices

    See Also
    --------
    skeleton_refinement.stochastic_registration.perform_registration : Core function used to refine the skeleton
    plant3dvision.tasks.proc3d.PointCloud : Task that provides the point cloud data
    plant3dvision.tasks.proc3d.CurveSkeleton : Task that generates the initial skeleton

    Notes
    -----
    The refinement process involves two main steps:
    1. Stochastic deformation registration to adjust skeleton points to better match the point cloud
    2. Optional connectivity reconstruction using minimum spanning tree on the k-nearest neighbor graph

    If `knn_mst` is False, the original skeleton connectivity is preserved while using the refined point positions.
    """
    upstream_task = luigi.TaskParameter(default=CurveSkeleton)  # override default attribute from ``RomiTask``
    upstream_pcd = luigi.TaskParameter(default=PointCloud)
    alpha = luigi.FloatParameter(default=5.)
    beta = luigi.FloatParameter(default=5.)
    max_iterations = luigi.IntParameter(default=100)
    tolerance = luigi.FloatParameter(default=0.0001)
    knn_mst = luigi.BoolParameter(default=True)
    n_neighbors = luigi.IntParameter(default=5)
    knn_algorithm = luigi.Parameter(default='kd_tree')  # 'auto', 'ball_tree', 'kd_tree' or 'brute'.
    mst_algorithm = luigi.Parameter(default='kruskal')  # 'kruskal', 'prim' or 'boruvka'.

    def requires(self):
        return {"skeleton": self.upstream_task(), "pcd": self.upstream_pcd()}

    def run(self):
        from skeleton_refinement.stochastic_registration import perform_registration
        # Read input skeleton from JSON file (contains points and lines)
        skel = io.read_json(self.input()["skeleton"].get().get_file("CurveSkeleton"))
        # Read input point cloud data
        pcd = io.read_point_cloud(self.input()["pcd"].get().get_file("PointCloud"))

        # Perform stochastic registration to refine skeleton points
        # Uses point cloud and skeleton points as input, returns refined points
        refined_skel = perform_registration(np.asarray(pcd.points), np.array(skel["points"]),
                                            alpha=self.alpha, beta=self.beta,
                                            max_iterations=self.max_iterations,
                                            tolerance=self.tolerance)

        if self.knn_mst:
            # Create minimum spanning tree from refined skeleton using k-nearest neighbors
            skel_tree = knn_mst(refined_skel,
                                n_neighbors=int(self.n_neighbors),
                                knn_algorithm=str(self.knn_algorithm),
                                mst_algorithm=str(self.mst_algorithm))
            # Convert tree to points and lines format
            refined_skel = {
                "points": [skel_tree.nodes[node]['position'].tolist() for node in skel_tree.nodes],
                "lines": list(skel_tree.edges),
            }
        else:
            # Keep original connectivity (lines) with refined points
            refined_skel = {"points": refined_skel.tolist(), "lines": skel['lines']}

        # Write refined skeleton to JSON output file
        io.write_json(self.output_file(create=True), refined_skel)


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
        # Get the first file from upstream task's output (NPZ voxel volume)
        prediction_file = self.upstream_task().output().get().get_files()[0]
        # Read the NPZ file containing voxel data for different labels
        voxels = io.read_npz(prediction_file)

        # Initialize dictionary to store likelihood ratios for each label
        out = {}
        labels = list(voxels.keys())

        for label in labels:
            # Skip labels not present in both recall and specificity dictionaries
            if label in self.recall:
                recall = self.recall[label]
            else:
                continue
            if label in self.specificity:
                specificity = self.specificity[label]
            else:
                continue
            # Calculate log-likelihood for null hypothesis (H0)
            # H0: voxel doesn't belong to the class (using specificity)
            l0 = (self.n_views - voxels[label]) * np.log(specificity) + voxels[label] * np.log(1 - specificity)
            # Calculate log-likelihood for alternative hypothesis (H1)
            # H1: voxel belongs to the class (using recall)
            l1 = (self.n_views - voxels[label]) * np.log(1 - recall) + voxels[label] * np.log(recall)
            # Store log-likelihood ratio (L1/L0) for this label
            out[label] = l1 - l0  # FIXME ?

        # Create output file and save results
        outfile = self.output_file(create=True)
        io.write_npz(outfile, out)
        # Copy metadata from input file to output file
        outfile.set_metadata(prediction_file.get_metadata())
