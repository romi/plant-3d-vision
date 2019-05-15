# from romiscan.cl import ,FIM
from matplotlib import pyplot as plt
import numpy as np
# import fim
# import skfmm
import pyolim
import open3d

spd = np.load("/data/twintz/scanner/v0.4/arabidopsis062/BackProjection_True_1_0_7b8349e869/voxels.npy")
# spd = np.exp(spd)
# spd[spd < 0.0001] = 0

spd = spd - spd.min()
spd = spd / spd.max()
spd[spd < 0.3] = 0.00

seed = np.unravel_index(np.argmax(spd[:,:,0]), spd.shape[0:2])

bm = pyolim.Olim3dHuRect(1/spd)

# bm.add_boundary_node(0,seed[1], seed[0])
# 
bm.add_boundary_node(*seed, 0)
x,y,z = np.nonzero(spd == 0)
# for i in range(len(x)):
#     print("%i, %i, %i"%(x[i], y[i], z[i]))
#     bm.add_boundary_node(x[i], y[i], z[i], np.inf)

bm.run()
dist = [[[bm.get_value(i,j,k) for i in range(spd.shape[0])] for j in range(spd.shape[1])] for k in range(spd.shape[2])]
dist = np.array(dist)

# spd[spd < 0.1] = 0
# spd = spd + 0.0001

# MAX_DIST = 100

# se = fim.StructuredEikonal(True)

# pad = np.mod(16  - np.mod(spd.shape, 16), 16)
# pad = [(0,x) for x in pad]
# spd_pad = np.pad(spd, pad, mode='constant')

# se.set_dims(*spd_pad.shape)
# se.set_iters_per_block(10)
# seed = [*seed, 0]
# seeds = [seed]
 

# for i in range(100):
#     print("i = %i"%i)
#     se.set_seeds(seeds)
#     se.set_speeds(spd_pad.tolist())
#     se.solve_eikonal()
#     dist = np.array(se.get_final_result())
#     x,y,z = np.nonzero(dist<30)
#     u = np.argmax([np.min([(x[u] - s[0])**2 + 
#                         (y[u] - s[1])**2 + (z[u] - s[2])**2 for s in seeds]) for u in range(len(x))])
#     seeds.append([x[u], y[u], z[u]])





# fim_constant = FIM(spd.shape, [0,0,0], 1.0, spd_constant, max_dist = MAX_DIST)

# threshold = np.quantile(spd, 0.99)
# seed = np.unravel_index(np.argmax(spd[:,:,0] > threshold), spd.shape[0:2])
# seed = [*seed, 0]

# seeds_array = np.zeros((1,3))
# seeds_array[0,:] = seed

# fim.set_seeds(seeds_array)
# fim.run()

# fim_constant.set_seeds(seeds_array)
# fim_constant.run()

# idx = fim.get_border_pts()
# x, y, z = np.unravel_index(idx, spd.shape)
# array = np.zeros((len(x), 3))
# array[:, 0] = x
# array[:, 1] = y
# array[:, 2] = z
# pcd = open3d.geometry.PointCloud()
# pcd.points = open3d.utility.Vector3dVector(array)

# idx = fim_constant.get_border_pts()
# x, y, z = np.unravel_index(idx, spd.shape)
# array = np.zeros((len(x), 3))
# array[:, 0] = x
# array[:, 1] = y
# array[:, 2] = z
# pcd2 = open3d.geometry.PointCloud()
# pcd2.points = open3d.utility.Vector3dVector(array)


# dst = fim_constant.get_distance_map()
# dst_border = dst.flatten()[idx]
# keypt_border_idx = np.argmax(dst_border)
# keypt_idx = idx[keypt_border_idx]


# dst[np.isinf(dst)] = 0
# plt.imshow(dst.max(0))
# plt.scatter(keypt_z, keypt_y)
# plt.show()
