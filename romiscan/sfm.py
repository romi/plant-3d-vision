from romidata import io
import numpy as np
from romidata import fsdb
from tqdm import tqdm
import open3d
import cv2
from romiscan import pyceres


class StructureFromMotion():
    def __init__(self, images_fileset, camera_matrix, dist_coefs):
        self.images_fileset = images_fileset
        self.camera_matrix = camera_matrix
        self.dist_coefs = dist_coefs

    def compute_features(self):
        """
        Computes features in a list of images.
        """
        siftp = None
        result = {}
        cv2.ocl.useOpenCL()
        detector = cv2.xfeatures2d.SURF_create()
        for f in tqdm(self.images_fileset.get_files()):
            img = cv2.UMat(io.read_image(f))
            kp, des = detector.detectAndCompute(img, None)
            result[f.id] = kp, des
        self.keypoints = result

    def compute_pair_lists(self, n):
        pcd = open3d.geometry.PointCloud()
        points = np.zeros((0, 3))
        point = np.zeros((1, 3))
        corres = {}

        try:
            for i, f in enumerate(self.images_fileset.get_files()):
                pos = f.get_metadata("pose")
                point[0, :] = pos[:3]
                points = np.vstack([point, points])
                corres[i] = f.id
        except:
            raise BaseException("Cannot find camera pose in image metadata.")

        pcd.points = open3d.utility.Vector3dVector(points)
        pcd_tree = open3d.geometry.KDTreeFlann(pcd)

        result = {}
        for i in tqdm(range(len(pcd.points))):
            pos = f.get_metadata("pose")
            k, idx, _ = pcd_tree.search_knn_vector_3d(points[i, :], n)
            pair_list = np.array(idx, dtype=int).tolist()
            result[corres[i]] = [corres[j] for j in pair_list]
        self.pair_lists = result

    def compute_matches(self):
        result = {}
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        for f in tqdm(self.images_fileset.get_files()):
            ids = self.pair_lists[f.id]
            for m_id in ids:
                pair = (f.id, m_id)
                if f.id != m_id and pair not in result and (pair[1], pair[0]) not in result:
                    des1 = self.keypoints[pair[0]][1]
                    des2 = self.keypoints[pair[1]][1]
                    matches = flann.knnMatch(des1,des2,k=2)
                    result[pair] = matches
        self.matches = result

    def filter_matches(self, threshold=0.9):
        result = {}
        for p in tqdm(self.matches.keys()):
            pair_matches = self.matches[p]
            pts1 = []
            pts2 = []

            kpts_1, _ = self.keypoints[p[0]]
            kpts_2, _ = self.keypoints[p[1]]

            for m in pair_matches:
                if m[0].distance > threshold * m[1].distance:
                    continue
                idx_1 = m[0].queryIdx
                idx_2 = m[0].trainIdx
                pt_1 = kpts_1[idx_1].pt
                pt_2 = kpts_2[idx_2].pt

                pts1.append(pt_1)
                pts2.append(pt_2)

            pts1 = np.float32(pts1)
            pts2 = np.float32(pts2)

            focal = self.camera_matrix[0, 0]
            principal_point = (
                self.camera_matrix[0, 2], self.camera_matrix[1, 2])
            # principal_point = np.int32(principal_point)

            E, mask = cv2.findEssentialMat(
                pts1, pts2, focal=focal, pp=principal_point)

            retval, R, t, mask = cv2.recoverPose(
                E, pts1, pts2, self.camera_matrix)

            if mask.sum() > 0:
                result[p] = {
                    "rot": R,
                    "tvec": t,
                    "pts1": pts1[mask.ravel() > 0],
                    "pts2": pts2[mask.ravel() > 0]
                }

        self.filtered_matches = result

    def get_view_pairs(self, view):
        return {m : self.filtered_matches[m] for m in self.filtered_matches.keys() if view == m[0] or view == m[1]}

    def add_views(self):
        """
        """
        self.ba_view_index = {}
        self.initial_poses = {}

        first_view = list(self.keypoints.keys())[0]
        rot = np.eye(3)
        tvec = np.zeros(3)
        self.initial_poses[first_view] = rot, tvec

        view = first_view
        while True:
            pairs = self.get_view_pairs(view)
            pairs = {k: v for k, v in pairs.items() if not (k[0] in self.initial_poses and k[1] in self.initial_poses)}

            if len(pairs) == 0:
                break

            best_pair = max(pairs, key=lambda k: len(
                self.filtered_matches[k]["pts1"]))
            if view == best_pair[0]:
                new_view = best_pair[1]
                rot = self.filtered_matches[best_pair]["rot"]
                tvec = self.filtered_matches[best_pair]["tvec"].ravel()
            else:
                new_view = best_pair[0]
                rot = self.filtered_matches[best_pair]["rot"].transpose()
                tvec = -np.dot(rot, self.filtered_matches[best_pair]["tvec"]).ravel()

            rot_1, tvec_1 = self.initial_poses[view]
            rot_2 = np.dot(rot, rot_1)
            tvec_2 = np.dot(rot, tvec_1) + tvec
            self.initial_poses[new_view] = rot_2, tvec_2.ravel()
            view = new_view

        for k in self.initial_poses.keys():
            rot, tvec = self.initial_poses[k]
            idx = self.ba.add_view(rot.ravel().tolist(), tvec.ravel().tolist())
            self.ba_view_index[k] = idx

    def add_pair(self, pair):
        """
        Add a pair to the bundle adjustment function.
        """

        rot_1, tvec_1 = self.initial_poses[pair[0]]
        rot_2, tvec_2 = self.initial_poses[pair[1]]

        rot = rot_2*rot_1.transpose()
        tvec = tvec_2 - np.dot(rot_1.transpose(), tvec_1)

        u_rot = cv2.UMat(rot)
        u_tvec = cv2.UMat(tvec)

        R1, R2, P1, P2, Q, _, _= cv2.stereoRectify(cv2.UMat(self.camera_matrix), cv2.UMat(self.dist_coefs), cv2.UMat(self.camera_matrix), cv2.UMat(self.dist_coefs), (1920, 1080),
                u_rot,
                u_tvec)

        pts1 = self.filtered_matches[pair]["pts1"].transpose()
        pts2 = self.filtered_matches[pair]["pts2"].transpose()

        pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2).get()

        pts3d = np.zeros((3, pts4d.shape[1]))

        pts3d[0,:] = pts4d[0,:] / pts4d[3,:]
        pts3d[1,:] = pts4d[1,:] / pts4d[3,:]
        pts3d[2,:] = pts4d[2,:] / pts4d[3,:]

        pts3d = np.dot(rot, pts3d) + tvec[:, np.newaxis]

        idx_1 = self.ba_view_index[pair[0]]
        idx_2 = self.ba_view_index[pair[1]]
        self.ba.add_pair(idx_1, idx_2, pts1.ravel().tolist(),pts2.ravel().tolist(),pts3d.ravel().tolist())



    def bundle_adjustment(self):
        self.ba = pyceres.BundleAdjustment()
        intrinsics = [self.camera_matrix[0, 0], self.camera_matrix[0,2], self.camera_matrix[1,2], 0, 0]
        self.ba.set_intrinsics(intrinsics)

        self.add_views()

        for pair in self.filtered_matches.keys():
            self.add_pair(pair)

        self.ba.init_cost_function()
        self.ba.solve()

    def get_poses(self):
        l = self.ba.get_extrinsics()
        res = {}
        for k in self.keypoints.keys():
            idx = self.ba_view_index[k]
            a, tvec = l[idx]
            rot, _= cv2.Rodrigues(cv2.UMat(np.array(a)))
            res[k] = rot.get(), np.array(tvec)

        return res
        

if __name__ == "__main__":
    db = fsdb.FSDB("/data/twintz/scanner/stitch/")
    db.connect()
    scan = db.get_scan("light")
    images = scan.get_fileset("images")
    camera_matrix = np.array([[1379.78039550781, 0, 978.726440429688],
                              [0.0, 1379.78039550781, 529.610412597656],
                              [0.0, 0.0, 1.0]])
    dist_coefs = np.array([0,0,0,0,0])
    sfm = StructureFromMotion(images, camera_matrix, dist_coefs)
    sfm.compute_pair_lists(6)
    sfm.compute_features()
    sfm.compute_matches()
    sfm.filter_matches()
    sfm.bundle_adjustment()
    poses = sfm.get_poses()
    x = []
    y = []
    for k in poses.keys():
        rot, tvec = poses[k]
        pos = -np.dot(rot.transpose(), tvec)

        x.append(pos[0])
        y.append(pos[1])

    db.disconnect()
