#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:14:31 2019

@author: alienor
"""
import numpy as np
import os
import random
import json
from romidata import fsdb, io
from subprocess import call

blender_path = '/home/alienor/Documents/blender_virtual_scanner/'

target_dir = blender_path + 'data/COSEG/guitar/'
param = '/home/alienor/Documents/scanner-meta-repository/Scan3D/config/virtual_scanner.json'
objs = os.listdir(target_dir)
backgrounds = os.listdir(blender_path + '/hdri')
with open(param, "r") as jsonFile:
    data = json.load(jsonFile)


for obj in objs:
    if 'materials' not in obj:
        data["Scan"]["scanner"]["camera_args"]["load_object"] = obj + '?dx=%d&dy=%d&dz=%d'%(random.randint(-5, 5),
            random.randint(-5, 5),random.randint(-5, 5))
        data["Scan"]["scanner"]["camera_args"]["load_background"] = backgrounds[random.randint(0, len(backgrounds)-1)]
    
        with open(param, "w") as jsonFile:
            json.dump(data, jsonFile, ensure_ascii=False)
        
        color = np.random.rand(3)
        text = 'Kd %f %f %f\n'%tuple(color)
        with open(target_dir + 'materials.mtl', 'r') as file:
            props = file.readlines()
        pos = 6
        while pos + 3 < len(props):            
            props[pos] = text
            pos += 10
            
        with open(target_dir + 'materials.mtl', 'w') as file:
            file.writelines(props)
            
        #os.system('run-task --config /home/alienor/Documents/scanner-meta-repository/Scan3D/default/virtual_scanner.json Scan ../../training2D/data/database/' + obj[:-7] + ' --local-scheduler')
        database = '/home/alienor/Documents/training2D/data/database_poses/'
        scan_name = obj[:-7]
        path_obj =  database + obj[:-7]
        call(["run-task", "--config", "config/virtual_scanner.json", "Scan", path_obj, "--local-scheduler"])

        