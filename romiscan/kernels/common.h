int3 unravel_index(int idx, __global int *shape) {
    int nx = shape[0];
    int ny = shape[1];
    int nz = shape[2];

    int i = idx / (ny * nz);
    int j = (idx - i * ny * nz) / nz;
    int k = idx - i * ny * nz - j * nz;

    int3 res = {i, j, k};
    return res;
}

int ravel_index(int x, int y, int z, __global int *shape) {
    int nx = shape[0];
    int ny = shape[1];
    int nz = shape[2];
    return x * nz * ny + y * nz + z;
}

