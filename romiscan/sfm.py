from romidata import io
import numpy as np
from romidata import fsdb
from tqdm import tqdm
import open3d
import cv2
from romiscan import pyceres


class StructureFromMotion():
    def __init__(self, images_fileset, camera_matrix):
        self.images_fileset = images_fileset
        self.camera_matrix = camera_matrix

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
            keypoints = detector.detect(img)
            result[f.id] = keypoints
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
        siftp = sift.MatchPlan(devicetype="ALL")
        for f in tqdm(self.images_fileset.get_files()):
            ids = self.pair_lists[f.id]
            for m_id in ids:
                pair = (f.id, m_id)
                if f.id != m_id and pair not in result and (pair[1], pair[0]) not in result:
                    commonkp = siftp.match(
                        self.keypoints[f.id], self.keypoints[m_id])
                    # FLANN parameters
                        # FLANN_INDEX_KDTREE = 1
                        # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
                        # search_params = dict(checks=50)   # or pass empty dictionary
                        # flann = cv.FlannBasedMatcher(index_params,search_params)
                        # matches = flann.knnMatch(des1,des2,k=2)
                    result[pair] = commonkp
        self.matches = result

    def filter_matches(self):
        result = {}
        for p in tqdm(self.matches.keys()):
            pair_matches = self.matches[p]
            pts1 = []
            pts2 = []
            for m in pair_matches:
                pts1.append(np.array([m[0][0], m[0][1]]))
                pts2.append(np.array([m[1][0], m[1][1]]))

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

            pair_matches = pair_matches[mask.ravel() == 1]
            if mask.sum() > 0:
                result[p] = {
                    "rot": R,
                    "tvec": t,
                    "matches": pair_matches
                }

        self.filtered_matches = result

    def bundle_adjustment(self):
        initial_pair = max(self.filtered_matches, key=lambda k: len(
            self.filtered_matches[k]["matches"]))


if __name__ == "__main__":
    db = fsdb.FSDB("/data/twintz/scanner/stitch")
    db.connect()
    scan = db.get_scan("light")
    images = scan.get_fileset("images")
    camera_matrix = np.array([[1379.78039550781, 0, 978.726440429688],
                              [0.0, 1379.78039550781, 529.610412597656],
                              [0.0, 0.0, 1.0]])
    sfm = StructureFromMotion(images, camera_matrix)
    sfm.compute_pair_lists(3)
    sfm.compute_features()
    sfm.compute_matches()
    sfm.filter_matches()
    db.disconnect()
