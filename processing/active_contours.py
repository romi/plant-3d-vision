#!/usr/bin/env python3
# import autograd.numpy as np
import numpy as np
import sys,os
import open3d
from scipy.ndimage.filters import convolve
from scipy.interpolate import RegularGridInterpolator

class TubularActiveContours():
    def __init__(self, n_radius, width, radius, lam=1e-1):
        filter_size = int(radius/width) + 1

        # Create filters
        x,y,z = np.ogrid[-filter_size:filter_size+1,-filter_size:filter_size+1,-filter_size:filter_size+1]
        filter = x**2+y**2+z**2 <= (filter_size-1)**2
        self.filter = np.asarray(filter / (filter_size-1)**3, dtype=float)

        gx, gy, gz = np.gradient(self.filter)
        self.filter_grad = gx, gy, gz

        self.data_loaded = False
        self.lam = lam
        self.width = width

    def load_data(self, data):
        for i in range(3):
            data[:,i] = data[:,i] - np.min(data[:, i])

        indices = np.array(np.round(data[:, 0:3] / self.width), dtype=np.int)
        shape = indices.max(axis=0)
        A = np.zeros(shape+1, dtype=np.float)

        for i in range(data.shape[0]):
              A[indices[i,0], indices[i, 1], indices[i, 2]] = 1.

        # Create filtered volumes
        vol = np.zeros(A.shape)
        vol_grad = np.zeros((3, *A.shape))
        convolve(A, self.filter, mode='constant', output=vol)
        for k in range(3):
            convolve(A, self.filter_grad[k], mode='constant', output=vol_grad[k])

        # Create interpolator
        self.xmin, self.xmax = np.min(data[:, 0]), np.max(data[:, 0])
        self.ymin, self.ymax = np.min(data[:, 1]), np.max(data[:, 1])
        self.zmin, self.zmax = np.min(data[:, 2]), np.max(data[:, 2])

        x_values = np.linspace(self.xmin, self.xmax, A.shape[0])
        y_values = np.linspace(self.ymin, self.ymax, A.shape[1])
        z_values = np.linspace(self.zmin, self.zmax, A.shape[2])

        self.phi = RegularGridInterpolator((x_values, y_values, z_values), vol)
        self.dphidx = RegularGridInterpolator((x_values, y_values, z_values), vol_grad[0])
        self.dphidy = RegularGridInterpolator((x_values, y_values, z_values), vol_grad[1])
        self.dphidz = RegularGridInterpolator((x_values, y_values, z_values), vol_grad[2])

    def length(self, x):
        val = x[:-1, :] - x[1:, :]
        return np.linalg.norm(val, axis=1).sum()

    def tan(self, x):
        output = np.zeros(x.shape)
        y = x[:-2, :] - x[2:, :]
        y = y/np.linalg.norm(y, axis=1)[:, np.newaxis]
        output[1:-1,:] = y
        return output

    def gradlength(self, x):
        output = np.zeros(x.shape)
        val1 = x[1:-1, :] - x[:-2, :]
        val1 = val1 / np.linalg.norm(val1, axis=1)[:, np.newaxis]
        val2 = x[1:-1, :] - x[2:, :]
        val2 = val2 / np.linalg.norm(val2, axis=1)[:, np.newaxis]
        output[1:-1, :] = val1 + val2
        return output

    def V(self, x):
        values = -self.phi(x) + self.lam * self.length(x)
        return values.sum()

    def gradV(self, x):
        dx = -self.dphidx(x)
        dy = -self.dphidy(x)
        dz = -self.dphidz(x)
        res = np.vstack([dx, dy, dz]).transpose() + self.lam * self.gradlength(x)
        t = self.tan(x)
        res = res - t * np.sum(res*t, axis=1)[:, np.newaxis]
        res[0, :] = 0
        res[-1, :] = 0
        return res

    def step(self, x, coef):
        g = self.gradV(x)
        x -= coef*g
        x[:, 0] = np.maximum(x[:, 0], self.xmin)
        x[:, 0] = np.minimum(x[:, 0], self.xmax)
        x[:, 1] = np.maximum(x[:, 1], self.ymin)
        x[:, 1] = np.minimum(x[:, 1], self.ymax)
        x[:, 2] = np.maximum(x[:, 2], self.zmin)
        x[:, 2] = np.minimum(x[:, 2], self.zmax)
        return x

