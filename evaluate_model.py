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
import matplotlib.pyplot as plt
from romiseg.utils.alienlab import create_folder_if

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
save_dir = db_path+ '/Evaluation/'
create_folder_if(save_dir)
db.disconnect()

classes = [
        "flower",
        "fruit",
        "leaf",
        "pedicel",
        "stem"
    ]

bins = 100

histograms = {}
tasks_eval = ['Segmentation2DEvaluation', 'VoxelsEvaluation']#, 'PointCloudEvaluation']

# Initialisation

for c in classes:
    histograms[c] = {}
    for task_eval in tasks_eval:
        histograms[c][task_eval] = {}
        histograms[c][task_eval]['hist_high'] = np.zeros(bins)
        histograms[c][task_eval]['hist_low'] = np.zeros(bins)
        histograms[c][task_eval]['bins_high'] = np.zeros(bins + 1)
        histograms[c][task_eval]['bins_low'] = np.zeros(bins + 1)
            
# Iteration over the scans
for scan_id in scans:  
   if 'model' not in scan_id and 'Evaluation' not in scan_id:         
        # call(["romi_run_task", "--log-level", "WARNING", "--config", "/home/alienor/Documents/scanner-meta-repository/Scan3D/config/segmentation2d_arabidopsis.toml", "--module", 'romiscan.tasks.evaluation', task_eval, os.path.join(db_path, scan_id), "--local-scheduler"], check=True)
        db = fsdb.FSDB(db_path)
        db.connect()
        scan = db.get_scan(scan_id)
        for task_eval in tasks_eval:
            evaluation = scan.get_fileset(task_eval)
            f = evaluation.get_files()[0]
            results = io.read_json(f)
    
            for c in classes:
                if c in results.keys():
                    histograms[c][task_eval]['hist_high'] += results[c]['hist_high']
                    histograms[c][task_eval]['hist_low'] += results[c]['hist_low']
                    histograms[c][task_eval]['bins_high'] = results[c]['bins_high']
                    histograms[c][task_eval]['bins_low'] = results[c]['bins_low']
    
        
        db.disconnect()
        break
for task_eval in tasks_eval:        
   for c in classes:
      plt.figure()
      hist_high = histograms[c][task_eval]['hist_high']/np.sum(histograms[c][task_eval]['hist_high'])
      hist_low = histograms[c][task_eval]['hist_low']/np.sum(histograms[c][task_eval]['hist_low'])
      plt.plot(histograms[c][task_eval]['bins_high'][:-1], hist_high, label = c)
      plt.legend()
      plt.title('Histogram +')
      plt.xlabel('prediction level')
      plt.ylabel('num of hits')   

      plt.plot(histograms[c][task_eval]['bins_low'][:-1], hist_low, label = c)     
      plt.legend(['+', '-'])
      plt.title('Histogram -')
      plt.xlabel('prediction level')
      plt.ylabel('num of hits')
      plt.savefig(save_dir + "low_%s_%s.jpg"%(c, task_eval))

precision = {}
recall = {}
perfs = {}

for task_eval in tasks_eval:
    precision[task_eval] = {}
    recall[task_eval] = {}
    perfs[task_eval] = {}

plt.figure()

colors = ['r', 'g', 'b', 'k', 'c', 'm', '#eeefff']
for task_eval in tasks_eval:
   for i, c in enumerate(classes):
       precision_c = [] #0
       recall_c = [] #1
       baseline_c = []

       perfs[task_eval][c] = {}


       for lim in range(0, bins + 1):
           
              true_positive = np.sum(histograms[c][task_eval]['hist_high'][lim:])
              false_negative = np.sum(histograms[c][task_eval]['hist_high'][0:lim])
              true_negative = np.sum(histograms[c][task_eval]['hist_low'][0:lim])
              false_positive = np.sum(histograms[c][task_eval]['hist_low'][lim:])
              precision_c.append(true_positive/(true_positive + false_positive))
              recall_c.append(true_positive/(true_positive + false_negative))
              baseline_c.append(np.sum(histograms[c][task_eval]['hist_high'])/(np.sum(histograms[c][task_eval]['hist_high'])+np.sum(histograms[c][task_eval]['hist_low'])))

       precision[task_eval][c] = precision_c
       recall[task_eval][c] = recall_c
      
       plt.plot(precision[task_eval][c], recall[task_eval][c], '-', label = c, color = colors[i])
       plt.plot(histograms[c][task_eval]['bins_high'], baseline_c, '--', label = c, color = colors[i]) 
      
       area_PR = simps(precision[task_eval][c], recall[task_eval][c] )
       area_baseline = simps(baseline_c, histograms[c][task_eval]['bins_high'])

       perfs[task_eval][c]['AUC'] = area_PR - area_baseline
       perfs[task_eval][c]['precision@0.5'] = precision_c[len(precision_c)//2]
       perfs[task_eval][c]['recall@0.5'] = recall_c[len(recall_c)//2]
   
       if i == 0 or i == 5:
           plt.xlabel("recall")
           plt.ylabel("precision")
           plt.legend()
           plt.savefig(save_dir + 'precision_recall_%s_%s'%(c, task_eval))
           plt.figure()

with open(save_dir + 'perfs.txt', 'w') as outfile:
    json.dump(perfs, outfile)

x = np.arange(len(classes))  # the label locations
width = 0.35  # the width of the bars

prec_2 = [perfs[tasks_eval[0]][c]['precision@0.5'] for c in classes]
prec_3v = [perfs[tasks_eval[1]][c]['precision@0.5'] for c in classes]
#prec_3p = [perfs[tasks_eval[2]][c]['precision@0.5'] for c in classes]

rec_2 = [perfs[tasks_eval[0]][c]['recall@0.5'] for c in classes]
rec_3v = [perfs[tasks_eval[1]][c]['recall@0.5'] for c in classes]
#rec_3p = [perfs[tasks_eval[2]][c]['recall@0.5'] for c in classes]

fig, ax = plt.subplots()
y = width/3
rects1 = ax.bar(x - 3*y, prec_2, width/3, label='Segmentation precision', color = 'c')
rects2 = ax.bar(x - 2*y, prec_3v, width/3, label='Voxels precision', color = 'b')
#rects3 = ax.bar(x - y, prec_3p, width/3, label='Point cloud precision')

rects4 = ax.bar(x , rec_2, width/3, label='Segmentation recall', color = 'm')
rects5 = ax.bar(x + y, rec_3v, width/3, label='Voxels recall', color = 'r')
#rects6 = ax.bar(x + 2*y, rec_3p, width/3, label='Point Cloud recall')



# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Precision')

ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()

'''
def autolabel(rects):
    #Attach a text label above each bar in *rects*, displaying its height.
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
'''

fig.tight_layout()
fig.savefig(save_dir + 'compare_2D_3D')
