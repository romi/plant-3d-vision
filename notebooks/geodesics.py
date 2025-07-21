#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from plant3dvision.cl import FIM
from matplotlib import pyplot as plt
import numpy as np
# import pymrt
# import ogrid
# import fim
# import skfmm
# import pyolim
import open3d


spd = np.load("/data/twintz/scanner/v0.4/arabidopsis062/BackProjection_True_1_0_7b8349e869/voxels.npy")
# spd = np.exp(spd)
# spd[spd < 0.0001] = 0

spd = spd - spd.min()
spd = spd / spd.max()
spd = spd + 0.01

def dist2(x,y,z,pt):
    return (x-pt[0])**2 + (y-pt[1])**2 + (z-pt[2])**2

mx,my,mz = np.mgrid[0:spd.shape[0],0:spd.shape[1],0:spd.shape[2]]

# spd[spd < 0.01] = -1

seed = np.unravel_index(np.argmax(spd[:,:,0]), spd.shape[0:2])
seed = [*seed, 0]
seeds_array = np.zeros((1,3))
seeds_array[0,:] = seed

radius = 200
radius_min = 35


for i in range(30):
    print("i = %i"%i)
    ball = np.ones(spd.shape)
    for j in range(seeds_array.shape[0]):
        ball *= dist2(mx,my,mz,seeds_array[j,:]) > radius**2
    spd_masked = np.copy(spd)
    spd_masked[ball > 0] = -1.0

    fim = FIM(spd.shape, [0,0,0], 1.0, spd_masked)
    fim.set_seeds(seeds_array)
    fim.run()
    dst = fim.get_distance_map()
    del fim

    x,y,z = np.nonzero(dst < radius/2)
    m = np.inf
    idx = -1
    flag = False
    for j in range(seeds_array.shape[0]):
        D = dist2(x,y,z,seeds_array[j,:])
        u = np.argmax(D)
        print("j = %i, D = %i"%(j, np.sqrt(D[u])))
        if D[u] > radius_min**2:
            seeds_array = np.vstack([seeds_array, [x[u], y[u], z[u]]])
            flag = True

    if not flag:
        break
    # print("idx = %i, D = %i"%(j, np.sqrt(m)))

    dst[dst>radius/2] = radius


    plt.figure()
    plt.imshow(dst.min(0))
    plt.colorbar()
    plt.scatter(seeds_array[:,2], seeds_array[:,1])
    plt.savefig("test%i.png"%i)
    plt.close()

