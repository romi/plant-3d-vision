import cv2
import luigi
import numpy as np
import os


from romiscan.tasks import RomiTask, DatabaseConfig, FilesetTarget
from romiscan.tasks.colmap import Colmap
from romiscan.masking import *

class Undistort(RomiTask):
    """
    Undistorts images using openCV
    """
    def requires(self):
        return Colmap()

    def run(self):
        scan = self.output().scan
        try:
            camera = scan.get_metadata()['computed']['camera_model']
        except:
            camera = scan.get_metadata()['scanner']['camera_model']

        if camera is None:
            raise Exception("Could not find camera model for space carving")

        input_fileset = FilesetTarget(
            DatabaseConfig().db_location, DatabaseConfig().scan_id, "images").get()

        output_fileset = self.output().get()
        try:
            for fi in input_fileset.get_files():
                img = fi.read_image()
                ext = os.path.splitext(fi.filename)[-1][1:]
                camera_params = camera['params']
                mat = np.matrix([[camera_params[0], 0, camera_params[2]],
                                 [0, camera_params[1], camera_params[3]],
                                 [0, 0, 1]])
                undistort_parameters = np.array(camera_params[4:])
                undistorted_data = cv2.undistort(img, mat, undistort_parameters)

                newfi = output_fileset.create_file(fi.id)
                newfi.write_image(ext, undistorted_data)
        except:
            scan.delete_fileset(output_fileset.id)


class Masking(RomiTask):
    """
    Class for binary masking
    """

    type = luigi.Parameter()
    params = luigi.DictParameter()

    def requires(self):
        return Undistort()

    def run(self):
        if self.type == "linear":
            coefs = self.params["coefs"]
            dilation = self.params["dilation"]

            def f(x):
                img = (coefs[0] * x[:, :, 0] + coefs[1] * x[:, :, 1] +
                       coefs[2] * x[:, :, 2]) > coefs[3]
                for i in range(dilation):
                    img = binary_dilation(img)
                return img
        elif self.type == "excess_green":
            threshold = self.params["threshold"]
            dilation = self.params["dilation"]

            def f(x):
                img = excess_green(x) > threshold
                for i in range(dilation):
                    img = binary_dilation(img)
                return img
        else:
            raise Exception("Unknown masking type")

        output_fileset = self.output().get()
        for fi in self.input().get().get_files():
            data = fi.read_image()
            data = np.asarray(data, float)/255
            mask = f(data)
            mask = 255*np.asarray(mask, dtype=np.uint8)
            newf = output_fileset.get_file(fi.id, create=True)
            newf.write_image('png', mask)

class SoftMasking(RomiTask):
    """
    Class for 2D -> 2D filtering with float output.
    """

    type = luigi.Parameter()
    params = luigi.DictParameter(default=None)

    def requires(self):
        return Undistort()

    def run(self):
        if self.type == "linear":
            coefs = self.params["coefs"]
            scale = self.params["scale"]

            def f(x):
                x = gaussian_filter(x, scale)
                img = (coefs[0] * x[:, :, 0] + coefs[1] * x[:, :, 1] +
                       coefs[2] * x[:, :, 2])
                return img
        elif self.type == "excess_green":
            scale = self.params["scale"]

            def f(x):
                img = gaussian_filter(x, scale)
                img = excess_green(img)
                for i in range(dilation):
                    img = binary_dilation(img)
                return img
        elif self.type == "vesselness":
            scale = self.params["scale"]
            f = lambda x: vesselness_2D(x[:,:,1], scale)
        else:
            raise Exception("Unknown masking type")

        output_fileset = self.output().get()
        for fi in self.input().get().get_files():
            data = fi.read_image()
            data = np.asarray(data, float)/255
            mask = f(data)
            mask = np.asarray(255*mask, dtype=np.uint8)
            newf = output_fileset.get_file(fi.id, create=True)
            newf.write_image('png', mask)
