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
import copy
from romidata import fsdb, io
from subprocess import run as call
import argparse

from scipy.integrate import simps


default_test_dir = "/home/alienor/Documents/training2D/data/dataset_vscan_constfocal/test/"

parser = argparse.ArgumentParser(description='Test directory')

parser.add_argument('--directory', dest='directory', default=default_test_dir,
                    help='test dir, default: %s'%default_test_dir)


args = parser.parse_args()
db_path = args.directory

db = fsdb.FSDB(db_path)
db.connect()
scans = [scan.id for scan in db.get_scans()]
db.disconnect()

eval_scan_id = 'Evaluation'
ignore_scan_ids = ['model'] + [eval_scan_id]

classes = [
        "flower",
        "fruit",
        "leaf",
        "pedicel",
        "stem"
    ]

bins = 100
eval = {}
tasks_eval = ['Segmentation2DEvaluation', 'PointCloudSegmentationEvaluation']#, 'PointCloudEvaluation']

# Initialisation
for c in classes:
    eval[c] = {}
    for task_eval in tasks_eval:
        eval[c][task_eval] = {
                    "tp" : 0,
                    "fp" : 0,
                    "tn" : 0,
                    "fn" : 0
                }
            
db = fsdb.FSDB(db_path)
db.connect()

# Iteration over the scans
for scan_id in scans:  
   if scan_id not in ignore_scan_ids:
        scan = db.get_scan(scan_id)
        for task_eval in tasks_eval:
            evaluation = scan.get_fileset(task_eval)
            if evaluation is None:
                continue
            f = evaluation.get_files()[0]
            results = io.read_json(f)
    
            for c in classes:
                if c in results.keys():
                    eval[c][task_eval]['tp'] += results[c]['tp']
                    eval[c][task_eval]['fp'] += results[c]['fp']
                    eval[c][task_eval]['tn'] += results[c]['tn']
                    eval[c][task_eval]['fn'] += results[c]['fn']

for task_eval in tasks_eval:
    for c in classes:
        try:
            eval[c][task_eval]["precision"] = eval[c][task_eval]["tp"]  /( eval[c][task_eval]["tp"] + eval[c][task_eval]["fp"])
            eval[c][task_eval]["recall"] = eval[c][task_eval]["tp"]  / (eval[c][task_eval]["tp"] + eval[c][task_eval]["fn"])
        except:
            continue

eval_scan = db.get_scan(eval_scan_id, create=True)
eval_fs = eval_scan.get_fileset(eval_scan_id, create=True)

eval_file = eval_fs.create_file("eval")
io.write_json(eval_file, eval)
