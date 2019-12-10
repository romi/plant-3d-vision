import luigi
import numpy as np

from romidata.task import  RomiTask, FileByFileTask
from romidata import io

from romiscan.tasks.scan import Scan
from romiscan.tasks.colmap import Colmap

class Undistorted(FileByFileTask):
    """Obtain undistorted images
    """
    upstream_task = luigi.TaskParameter(default=Scan)

    reader = io.read_image
    writer = io.write_image

    def input(self):
        return Scan().output()

    def requires(self):
        return [Colmap(), Scan()] 

    def f(self, x):
        from romiscan import proc2d
        scan = self.output().scan
        try:
            camera = scan.get_metadata()['computed']['camera_model']
        except:
            camera = scan.get_metadata()['scanner']['camera_model']

        if camera is None:
            raise Exception("Could not find camera model for undistortion")
        return proc2d.undistort(x, camera)

class Masks(FileByFileTask):
    """Mask images
    """
    upstream_task = luigi.TaskParameter(default=Undistorted)

    reader = io.read_image
    writer = io.write_image

    undistorted_input = luigi.BoolParameter(default=True)

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

    def f(self, x):
        from romiscan import proc2d
        x = self.f_raw(x)
        if self.binarize:
            x = x > self.threshold
            if self.dilation > 0:
                x = proc2d.dilation(x, self.dilation)
        else:
            x[x < self.threshold] = 0
            x = proc2d.rescale_intensity(x, out_range=(0, 1))
        x = np.array(255*x, dtype=np.uint8)
        return x
        
class Segmentation2D(RomiTask):
    """
    Segment images by class"""
    
    upstream_task = luigi.TaskParameter(default=Undistorted)

    label_names = luigi.Parameter(default=['background', 'flowers', 'peduncle', 'stem', 'leaves', 'fruits'])
    Sx = luigi.IntParameter(default=896)
    Sy = luigi.IntParameter(default=1000)
    model_segmentation_name = luigi.Parameter('ERROR')
    directory_weights = luigi.Parameter('complete here')

    single_label = luigi.Parameter(default=None)

    def requires(self):
        return self.upstream_image()


    def run(self):
        from romiseg.Segmentation2D import segmentation
        
        images_fileset = self.input_fileset()
        scan = images_fileset.scan
        
        
        #APPLY SEGMENTATION
        images_segmented, id_im = segmentation(self.Sx, self.Sy, self.label_names, 
                                        images_fileset, scan, self.model_segmentation_name, self.directory_weights)
        
        output_fileset = self.output().get()
        
        #Save prediction matrix [N_cam, N_labels, xinit, yinit]
        #f = output_fileset.create_file('full_prediction_matrix')
        #write_torch(f, images_segmented)
        #f.id = 'images_matrix'
        
        #Save class prediction as images, one by one, class per class
        print("Saving the segmented images, takes around 15 s")
        if self.single_label is None:
            for i in range(images_segmented.shape[0]):
                for j in range(len(self.label_names)):
                    f = output_fileset.create_file('%03d_%s'%(i, self.label_names[j]))
                    im = (images_segmented[i, j, :, :].cpu().numpy() * 255).astype(np.uint8)
                    io.write_image(f, im, 'png' )
                    orig_metadata = images_fileset.get_file(id_im[i][0]).get_metadata
                    f.set_metadata({'image_id' : id_im[i][0], 'label' : self.label_names[j], **orig_metadata})
        else:
            for i in range(images_segmented.shape[0]):
                j = self.label_names.index(self.single_label)
                f = output_fileset.create_file('%03d_%s'%(i, self.label_names[j]))
                im = (images_segmented[i, j, :, :].cpu().numpy() * 255).astype(np.uint8)
                io.write_image(f, im, 'png' )
                orig_metadata = images_fileset.get_file(id_im[i][0]).get_metadata
                f.set_metadata({'image_id' : id_im[i][0], 'label' : self.label_names[j], **orig_metadata})

        
