from romidata import io
import numpy as np
from romidata import fsdb
from silx.image import sift
from tqdm import tqdm
import open3d
import cv2


class StructureFromMotion():
    def __init__(self, images_fileset):
        self.images_fileset = images_fileset

    def compute_features(self):
        """
        Computes features in a list of images.
        """
        siftp = None
        cur_shape = None
        cur_dtype = None
        result = {}
        for f in tqdm(self.images_fileset.get_files()):
            img = io.read_image(f)
            if img.shape != cur_shape or img.dtype != cur_dtype:
                siftp = sift.SiftPlan(img.shape, img.dtype, devicetype="GPU")
                cur_shape = img.shape
            keypoints = siftp.keypoints(img)
            result[f.id] = keypoints
        self.keypoints = result

    def compute_pair_lists(self, n):
        pcd = open3d.geometry.PointCloud()
        points = np.zeros((0,3))
        point = np.zeros((1,3))
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
            k, idx,_ = pcd_tree.search_knn_vector_3d(points[i,:], n)
            pair_list =  np.array(idx, dtype=int).tolist()
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
                    commonkp = siftp.match(self.keypoints[f.id], self.keypoints[m_id])
                    result[pair] = commonkp
        self.matches = result

    def filter_matches(self):
        result = {}
        for p in tqdm(self.matches.keys()):
            pair_matches = self.matches[p]
            pts1 = []
            pts2 = []
            for m in pair_matches:
                pts1.append(np.array([m[0][0], m[0][1], m[0][2]]))
                pts2.append(np.array([m[1][0], m[1][1], m[1][2]]))

            pts1 = np.int32(pts1)
            pts2 = np.int32(pts2)

            F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
            pair_matches = pair_matches[mask.ravel() == 1]


            result[p] = F, pair_matches
        self.filtered_matches = result



if __name__ == "__main__":
    db = fsdb.FSDB("/data/twintz/scanner/stitch")
    db.connect()
    scan = db.get_scan("2019-04-16_15-15-09")
    images = scan.get_fileset("images")
    sfm = StructureFromMotion(images)
    sfm.compute_pair_lists(3)
    sfm.compute_features()
    sfm.compute_matches()
    sfm.filter_matches()
    camera_matrix = np.array([[1379.78039550781, 0, 978.726440429688],
                              [0.0, 1379.78039550781, 529.610412597656],
                              [0.0, 0.0, 1.0]])
    db.disconnect()
