#!/usr/bin/env python3

import sys

sys.path.append("/home/alienor/Documents/Segmentation/")

import luigi

from romidata.task import RomiTask
from romiscan import tasks
from romidata import io
import json
import torch
import numpy as np

import importlib
import os

from romidata.task import ImagesFilesetExists, RomiTask, FileByFileTask
from romidata import io

from romiscan.filenames import *
from romiscan import arabidopsis
from romiscan import proc2d
from romiscan import proc3d
from romiscan import cl
from romiscan import colmap
from romiscan.tasks import *
import open3d

from Segmentation2D import segmentation




#####
def read_torch(dbfile, ext="pt"):
    """Reads torch tensor from a DB file.
    Parameters
    __________
    dbfile : db.File

    Returns
    _______
    Torch.Tensor
    """
    b = dbfile.read_raw()
    with tempfile.TemporaryDirectory() as d:
        fname = os.path.join(d, "temp.%s"%ext)
        with open(fname, "wb") as fh:
            fh.write(b)
        return torch.load(fname)

def write_torch(dbfile, data, ext="pt"):
    """Writes point cloud to a DB file.
    Parameters
    __________
    dbfile : db.File
    data : TorchTensor
    ext : str
        file extension (defaults to "pt").
    """
    with tempfile.TemporaryDirectory() as d:
        fname = os.path.join(d, "temp.%s"%ext)
        torch.save(data, fname)

        dbfile.import_file(fname)
#####     
        
class Segmentation2D(RomiTask):
    """
    Segment images by class"""
    
    upstream_task = None
    upstream_image = luigi.TaskParameter(default=Undistorted)
    upstream_colmap = luigi.TaskParameter(default=Colmap)

    label_names = luigi.ListParameter(default=['background', 'flowers', 'peduncle', 'stem', 'leaves', 'fruits'])
    Sx = luigi.IntParameter(default=896)
    Sy = luigi.IntParameter(default=1000)
    model_segmentation_name = luigi.Parameter('ERROR')

    def requires(self):
        return {'images': self.upstream_image(), 'colmap': self.upstream_colmap()}


    def run(self):
        
        images_fileset = self.input()['images'].get().get_files()
        colmap_fileset = self.input()['colmap'].get()

        scan = colmap_fileset.scan
        
        
        #APPLY SEGMENTATION
        images_segmented = segmentation(self.Sx, self.Sy, self.label_names, 
                                        images_fileset, scan, self.model_segmentation_name)
        
        output_fileset = self.output().get()
        
        #Save prediction matrix [N_cam, N_labels, xinit, yinit]
        f = output_fileset.create_file('full_prediction_matrix')
        write_torch(f, images_segmented)
        f.id = 'images_matrix'
        
        #Save class prediction as images, one by one, class per class
        for i in range(images_segmented.shape[0]):
            for j in range(len(self.label_names)):
                f = output_fileset.create_file('%03d_%s'%(i, self.label_names[j]))
                im = (images_segmented[i, j, :, :].cpu().numpy() * 255).astype(np.uint8)
                io.write_image(f, im, 'png' )
                f.set_metadata({'image_id' : i, 'label' : self.label_names[j]})

        
if __name__ == "__main__":
    luigi.run()