import luigi
import logging
import numpy as np

from romidata.task import  RomiTask, FileByFileTask, ImagesFilesetExists, FilesetExists
from romidata import io
from romiscan.tasks.colmap import Colmap

logger = logging.getLogger('romiscan')

class Undistorted(FileByFileTask):
    """Obtain undistorted images
    """
    upstream_task = luigi.TaskParameter(default=ImagesFilesetExists)

    def input(self):
        return self.upstream_task().output()

    def requires(self):
        return [Colmap(), self.upstream_task()] 

    def f(self, fi, outfs):
        from romiscan import proc2d
        camera_model = fi.get_metadata('camera')['camera_model']

        x = io.read_image(fi)
        x = proc2d.undistort(x, camera_model)

        outfi = outfs.create_file(fi.id)
        io.write_image(outfi, x)
        return outfi

class Masks(FileByFileTask):
    """Mask images
    """
    upstream_task = luigi.TaskParameter(default=Undistorted)

    type = luigi.Parameter()
    parameters = luigi.ListParameter(default=[])
    dilation = luigi.IntParameter()

    binarize = luigi.BoolParameter(default=True)
    threshold = luigi.FloatParameter(default=0.0)

    def f_raw(self, x):
        from romiscan import proc2d
        x = np.asarray(x, dtype=np.float)
        x = proc2d.rescale_intensity(x, out_range=(0, 1))
        if self.type == "linear":
            coefs = self.parameters
            return (coefs[0] * x[:, :, 0] + coefs[1] * x[:, :, 1] +
                   coefs[2] * x[:, :, 2])
        elif self.type == "excess_green":
            return proc2d.excess_green(x)
        elif self.type == "vesselness":
            scale = self.parameters['scale']
            channel = self.parameters['channel']
            return proc2d.vesselness_2D(x, scale, channel=channel)
        elif self.type == "invert":
            return 1-x
        else:
            raise Exception("Unknown masking type")

    def f(self, fi, outfs):
        from romiscan import proc2d
        x = io.read_image(fi)
        x = self.f_raw(x)
        if self.binarize:
            x = x > self.threshold
            if self.dilation > 0:
                x = proc2d.dilation(x, self.dilation)
        else:
            x[x < self.threshold] = 0
            x = proc2d.rescale_intensity(x, out_range=(0, 1))
        x = np.array(255*x, dtype=np.uint8)

        outfi = outfs.create_file(fi.id)
        io.write_image(outfi, x)
        return outfi

class ModelFileset(FilesetExists):
    scan_id = luigi.Parameter()
    fileset_id = "models"


class Segmentation2D(RomiTask):
    """
    Segment images by class"""

    upstream_task = luigi.TaskParameter(default=Undistorted)
    model_fileset = luigi.TaskParameter(default=ModelFileset)

    model_id = luigi.Parameter()
    query = luigi.DictParameter(default={})

    Sx = luigi.IntParameter(default=896)
    Sy = luigi.IntParameter(default=896)

    single_label = luigi.Parameter(default="")
    resize = luigi.BoolParameter(default=False)

    def requires(self):
        return {
            "images" : self.upstream_task(),
            "model" : self.model_fileset()
        }


    def run(self):
        from romiseg.Segmentation2D import segmentation
        import appdirs
        from skimage import transform
        import PIL

        images_fileset = self.input()["images"].get().get_files(query=self.query)
        model_file = self.input()["model"].get().get_file(self.model_id)
        if model_file is None:
            raise IOError("unable to find model: %s"%self.model_id)

        labels = model_file.get_metadata("label_names")

        #APPLY SEGMENTATION
        images_segmented, id_im = segmentation(self.Sx, self.Sy, images_fileset, model_file, self.resize)
        output_fileset = self.output().get()

        #Save class prediction as images, one by one, class per class
        logger.debug("Saving the .astype(np.uint8)segmented images, takes around 15 s")
        if self.single_label == "":
            for i in range(images_segmented.shape[0]):
                for j in range(len(labels)):
                    f = output_fileset.create_file('%03d_%s'%(i, labels[j]))
                    im = (images_segmented[i, j, :, :].cpu().numpy() * 255).astype(np.uint8)
                    io.write_image(f, im, 'png' )
                    orig_metadata = images_fileset[i].get_metadata()
                    f.set_metadata({'image_id' : id_im[i][0], 'label' : labels[j], **orig_metadata})
        else:
            for i in range(images_segmented.shape[0]):
                j = labels.index(self.single_label)
                f = output_fileset.create_file('%03d_%s'%(i, labels[j]))
                im = (images_segmented[i, j, :, :].cpu().numpy() * 255).astype(np.uint8)
                io.write_image(f, im, 'png' )
                orig_metadata = images_fileset[i].get_metadata()
                f.set_metadata({'image_id' : id_im[i][0], 'label' : labels[j], **orig_metadata})

        
