import luigi
import logging
import numpy as np

from romidata.task import  RomiTask, FileByFileTask
from romidata import io

from romiscanner.scan import Scan
from romiscan.tasks.colmap import Colmap

logger = logging.getLogger('romiscan')

class Undistorted(FileByFileTask):
    """Obtain undistorted images
    """
    upstream_task = luigi.TaskParameter(default=Scan)

    def input(self):
        return Scan().output()

    def requires(self):
        return [Colmap(), Scan()] 

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
        

class Segmentation2D(RomiTask):
    """
    Segment images by class"""
    import appdirs    
    upstream_task = luigi.TaskParameter(default=Undistorted)
    query = luigi.DictParameter(default={})

    labels = luigi.Parameter(default='background,flowers,peduncle,stem,bud,leaves,fruits')
    Sx = luigi.IntParameter(default=896)
    Sy = luigi.IntParameter(default=1000)
    model_segmentation_name = luigi.Parameter('ERROR')
    directory_weights = luigi.Parameter(default = appdirs.user_cache_dir())

    single_label = luigi.Parameter(default="")
    resize = luigi.BoolParameter(default=False)

    def requires(self):
        return self.upstream_task()


    def run(self):
        from romiseg.Segmentation2D import segmentation
        from skimage import transform
        import PIL
        images_fileset = self.input().get().get_files(query=self.query)
        scan = self.input().scan
        self.label_names = self.labels.split(',')
        #APPLY SEGMENTATION
        images_segmented, id_im = segmentation(self.Sx, self.Sy, self.label_names, 
                                        images_fileset, scan, self.model_segmentation_name, self.directory_weights, self.resize)
        
        output_fileset = self.output().get()
        
        #Save prediction matrix [N_cam, N_labels, xinit, yinit]
        #f = output_fileset.create_file('full_prediction_matrix')
        #write_torch(f, images_segmented)
        #f.id = 'images_matrix'
        
        #Save class prediction as images, one by one, class per class
        logger.debug("Saving the .astype(np.uint8)segmented images, takes around 15 s")
        if self.single_label == "":
            for i in range(images_segmented.shape[0]):
                for j in range(len(self.label_names)):
                    f = output_fileset.create_file('%03d_%s'%(i, self.label_names[j]))
                    im = (images_segmented[i, j, :, :].cpu().numpy() * 255).astype(np.uint8)
                    io.write_image(f, im, 'png' )
                    orig_metadata = images_fileset[i].get_metadata()
                    f.set_metadata({'image_id' : id_im[i][0], 'label' : self.label_names[j], **orig_metadata})
        else:
            for i in range(images_segmented.shape[0]):
                j = self.label_names.index(self.single_label)
                f = output_fileset.create_file('%03d_%s'%(i, self.label_names[j]))
                im = (images_segmented[i, j, :, :].cpu().numpy() * 255).astype(np.uint8)
                io.write_image(f, im, 'png' )
                orig_metadata = images_fileset[i].get_metadata()
                f.set_metadata({'image_id' : id_im[i][0], 'label' : self.label_names[j], **orig_metadata})

        
