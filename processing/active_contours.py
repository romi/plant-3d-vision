#!/usr/bin/env python3
# import autograd.numpy as np
import numpy as np
import sys,os
import open3d
from scipy.ndimage.filters import convolve
from scipy.interpolate import RegularGridInterpolator

class TubularActiveContours():
    def __init__(self, n_radius, lam1=1e-5, lam2=1e-1):
        # Create filters
        self.filters = {}
        self.filter_grads = {}
        for i in range(1, n_radius+1):
            x,y,z = np.ogrid[-i:i+1,-i:i+1,-i:i+1]
            filter = x**2+y**2+z**2 <= (i-1)**2
            filter = np.asarray(filter / i**3, dtype=float)
            self.filters[i] = filter
            gx, gy, gz = np.gradient(filter)
            self.filter_grads[i] = gx, gy, gz
        self.data_loaded = False
        self.lam1 = lam1
        self.lam2 = lam2
        self.n_radius = n_radius

    def load_data(self, data, width):
        for i in range(3):
            data[:,i] = data[:,i] - np.min(data[:, i])

        indices = np.array(np.round(data[:, 0:3] / width), dtype=np.int)
        shape = indices.max(axis=0)
        A = np.zeros(shape+1, dtype=np.float)

        for i in range(data.shape[0]):
              A[indices[i,0], indices[i, 1], indices[i, 2]] = 1.

        # Create filtered volumes
        vol = np.zeros((self.n_radius, *A.shape))
        vol_grad = np.zeros((3, self.n_radius, *A.shape))
        for i in range(1, self.n_radius+1):
            convolve(A, self.filters[i], mode='constant', output=vol[i-1])
            for k in range(3):
                convolve(A, self.filter_grads[i][k], mode='constant', output=vol_grad[k, i-1])

        dr = np.gradient(vol, axis=0)

        # Create interpolator
        self.xmin, self.xmax = np.min(data[:, 0]), np.max(data[:, 0])
        self.ymin, self.ymax = np.min(data[:, 1]), np.max(data[:, 1])
        self.zmin, self.zmax = np.min(data[:, 2]), np.max(data[:, 2])
        self.rmin, self.rmax = width, self.n_radius*width

        x_values = np.linspace(self.xmin, self.xmax, A.shape[0])
        y_values = np.linspace(self.ymin, self.ymax, A.shape[1])
        z_values = np.linspace(self.zmin, self.zmax, A.shape[2])
        r_values = np.linspace(self.rmin, self.rmax, self.n_radius)

        self.phi = RegularGridInterpolator((r_values, x_values, y_values, z_values), vol)
        self.dphidx = RegularGridInterpolator((r_values, x_values, y_values, z_values), vol_grad[0])
        self.dphidy = RegularGridInterpolator((r_values, x_values, y_values, z_values), vol_grad[1])
        self.dphidz = RegularGridInterpolator((r_values, x_values, y_values, z_values), vol_grad[2])
        self.dphidr = RegularGridInterpolator((r_values, x_values, y_values, z_values), dr)

    def grad(self, x):
        val = x[:-1, 1:4] - x[1:, 1:4]
        return val

    def length(self, x):
        g = self.grad(x)
        return np.linalg.norm(g, axis=1).sum()

    def tan(self, x):
        output = np.zeros(x.shape)
        y = self.grad(x)
        y = y[1:] + y[:-1]
        y = y/np.linalg.norm(y, axis=1)[:, np.newaxis]
        output[1:-1,1:4] = y
        return output

    def gradlength(self, x):
        output = np.zeros(x.shape)
        val1 = x[1:-1, 1:4] - x[:-2, 1:4]
        val1 = val1 / np.linalg.norm(val1, axis=1)[:, np.newaxis]
        val2 = x[1:-1, 1:4] - x[2:, 1:4]
        val2 = val2 / np.linalg.norm(val2, axis=1)[:, np.newaxis]
        output[1:-1, 1:4] = val1 + val2
        return output

    def V(self, x):
        values = -self.phi(x) + self.lam1/x[:, 0] + self.lam2 * self.length(x)
        return values.sum()

    def gradV(self, x):
        dx = -self.dphidx(x)
        dy = -self.dphidy(x)
        dz = -self.dphidz(x)
        dr = -self.dphidr(x) - self.lam1/x[:, 0]**2
        res = np.vstack([dr, dx, dy, dz]).transpose() + self.lam2 * self.gradlength(x)
        t = self.tan(x)
        res = res - t * np.sum(res*self.tan(x), axis=1)[:, np.newaxis]
        return res

    def step(self, x, coef):
        descent = coef*self.gradV(x)
        x -= descent
        x[:, 0] = np.maximum(x[:, 0], self.rmin)
        x[:, 0] = np.minimum(x[:, 0], self.rmax)
        x[:, 1] = np.maximum(x[:, 1], self.xmin)
        x[:, 1] = np.minimum(x[:, 1], self.xmax)
        x[:, 2] = np.maximum(x[:, 2], self.ymin)
        x[:, 2] = np.minimum(x[:, 2], self.ymax)
        x[:, 3] = np.maximum(x[:, 3], self.zmin)
        x[:, 3] = np.minimum(x[:, 3], self.zmax)
        return x

