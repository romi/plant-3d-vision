import luigi
import numpy as np

from romidata.task import  RomiTask, FileByFileTask
from romidata import io

from romiscan.tasks.proc3d import CurveSkeleton, ClusteredPointCloud

class TreeGraph(RomiTask):
    """Computes a tree graph of the plant.
    """
    upstream_task = luigi.TaskParameter(default=CurveSkeleton)

    z_axis =  luigi.IntParameter(default=2)
    z_orientation =  luigi.IntParameter(default=1)

    def run(self):
        from romiscan import arabidopsis
        f = io.read_json(self.input_file())
        t = arabidopsis.compute_tree_graph(f["points"], f["lines"], self.z_axis, self.z_orientation)
        io.write_graph(self.output_file(), t)

class AnglesAndInternodes(RomiTask):
    """Computes angles and internodes from skeleton
    """
    upstream_task = luigi.TaskParameter(default=TreeGraph)

    z_orientation = luigi.Parameter(default="down")

    def run(self):
        from romiscan import arabidopsis
        t = io.read_graph(self.input_file())
        measures = arabidopsis.compute_angles_and_internodes(t)
        io.write_json(self.output_file(), measures)

class AltAnglesAndInternodes(RomiTask):
    """Computes angles and internodes from skeleton
    """
    upstream_task = luigi.TaskParameter(default=ClusteredPointCloud)

    characteristic_length = luigi.FloatParameter(default=3.0)
    number_nn = luigi.IntParameter(default=50)
    stem_axis = luigi.IntParameter(default=2)
    stem_axis_inverted = luigi.BoolParameter(default=False)

    def run(self):
        import open3d
        input_fileset = self.input().get()

        stem_meshes = [io.read_triangle_mesh(f) for f in input_fileset.get_files(query={"label": "stem"})]
        stem_mesh = open3d.geometry.TriangleMesh()
        for m in stem_meshes:
            stem_mesh = stem_mesh + m

        stem_points = np.asarray(stem_mesh.vertices)

        idx_min = np.argmin(stem_points[:, self.stem_axis])
        idx_max = np.argmax(stem_points[:, self.stem_axis])

        stem_axis_min = stem_points[idx_min, self.stem_axis]
        stem_axis_max = stem_points[idx_max, self.stem_axis]

        stem_frame_axis = np.arange(stem_axis_min, stem_axis_max, self.characteristic_length)
        stem_frame = np.zeros((len(stem_frame_axis), 3, 4))

        kdtree = open3d.geometry.KDTreeFlann(stem_mesh)

        point = stem_points[idx_min]
        test = []

        ls = open3d.geometry.LineSet()
        lines = [[i,i+1] for i in range(len(stem_frame_axis) - 1)]
        pts = np.zeros((len(stem_frame_axis), 3))
        prev_axis = np.eye(3)

        if self.stem_axis_inverted:
            prev_axis[self.stem_axis, self.stem_axis] = -1

        gs= []

        for i, axis in enumerate(stem_frame_axis):
            point[self.stem_axis] = axis
            k, idx, _ = kdtree.search_knn_vector_3d(point, 50)
            vtx = np.asarray(stem_mesh.vertices)[idx]
            mean = vtx.mean(axis=0)
            u,s,v = np.linalg.svd(vtx - mean)
            print(v[0,:])
            first_vector = v[0, :]
            if first_vector[self.stem_axis] < 0 and not self.stem_axis_inverted:
                first_vector = -first_vector
            elif first_vector[self.stem_axis] > 0 and self.stem_axis_inverted:
                first_vector = -first_vector

            second_vector = prev_axis[2] - np.dot(prev_axis[2], first_vector)*first_vector
            second_vector = second_vector / np.linalg.norm(second_vector)

            third_vector = np.cross(first_vector, second_vector)

            rot = np.array([first_vector, second_vector, third_vector])
            prev_axis = rot

            stem_frame[i][0:3, 0:3] = rot.transpose()
            stem_frame[i][:, 3] = np.dot(rot, mean)

            point = mean
            pts[i,:] = mean

            visu_trans = np.zeros((4,4))
            visu_trans[:3, :3] = rot.transpose()
            visu_trans[:3, 3] = mean
            visu_trans[3,3] = 1.0

            f = open3d.geometry.TriangleMesh.create_coordinate_frame(size=20)
            f.transform(visu_trans)
            gs.append(f)

        ls.points = open3d.utility.Vector3dVector(pts)
        ls.lines = open3d.utility.Vector2iVector(lines)

        open3d.visualization.draw_geometries([ls, *gs])

        peduncle_meshes = [io.read_triangle_mesh(f) for f in input_fileset.get_files(query={"label": "peduncle"})]
        fruit_meshes = [io.read_triangle_mesh(f) for f in input_fileset.get_files(query={"label": "fruit"})]

        for f in fruit_meshes:
            continue

