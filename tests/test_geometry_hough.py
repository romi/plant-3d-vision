from lettucescan.geometry import util, hough
from skimage.filters import gaussian
import open3d
import os
import numpy as np

pcd = open3d.read_point_cloud(os.path.join(os.path.dirname(__file__),
                "../data/2019-01-22_15-57-34/pointcloud/voxels.ply"))
pts = np.asarray(pcd.points)

print("PCD to voxels...")
A = util.pcd_to_voxels(pts, 0.5, 10)

print("Gaussian filter...")
A = gaussian(A, 10)


values_ax = np.linspace(-0.4, 0.4, 20)
values_bx = np.linspace(50, 150, 100)
values_ay = np.linspace(-0.4, 0.4, 20)
values_by = np.linspace(50, 150, 100)

output = np.zeros((len(values_ax), len(values_bx), len(values_ay), len(values_by)))

print("Hough transform...")
hough.hough_transform(A, output, values_ax, values_bx, values_ay, values_by)

x = output.argmax()
x = np.unravel_index(x, output.shape)

ax = values_ax[x[0]]
bx = values_bx[x[1]]

ay = values_ax[x[2]]
by = values_bx[x[3]]

print("ax = %.2f"%ax)
print("bx = %.2f"%bx)
print("ay = %.2f"%ay)
print("by = %.2f"%by)

mesh = open3d.create_mesh_cylinder(radius=10, height=200)
dir_vect = np.array([ax, ay, 1])
dir_vect = dir_vect / np.linalg.norm(dir_vect)
translation = np.array([bx, by, 0])

transformation_matrix = np.zeros((4,4))
np.fill_diagonal(transformation_matrix, 1.0)
transformation_matrix[0:3, 3] = translation

mesh = mesh.transform(transformation_matrix)

open3d.draw_geometries([pcd, mesh])

print("done. Goodbye!")

