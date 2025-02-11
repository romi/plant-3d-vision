/**
 * @file common.h
 * @brief Provides functions for converting between linear and 3D indices.
 *
 * The functions included here are often utilized in grid-based computations
 * where a single 'idx' refers to a location within a three-dimensional volume.
 */

/**
 * @brief Converts a linear index into 3D coordinates.
 *
 * This function computes the (x, y, z) coordinates corresponding to
 * a single index 'idx' in a 3D array defined by 'shape'. Each dimension
 * size is stored in shape[0], shape[1], and shape[2], respectively.
 *
 * @param idx   The linear index to be converted.
 * @param shape A pointer to an array of three integers, containing the
 *              sizes of the dimensions in the order [nx, ny, nz].
 * @return      A struct (int3) containing the x, y, and z indices.
 */
int3 unravel_index(int idx, __global int *shape) {
    int nx = shape[0];  // number of elements in the x dimension
    int ny = shape[1];  // number of elements in the y dimension
    int nz = shape[2];  // number of elements in the z dimension

    // Calculate the x-coordinate by dividing by the product of y and z dimensions
    int i = idx / (ny * nz);

    // Then, derive the y-coordinate by dividing the remainder by the z dimension
    int j = (idx - i * ny * nz) / nz;

    // Finally, compute the z-coordinate from the final remainder
    int k = idx - i * ny * nz - j * nz;

    // Package the results into an int3 struct
    int3 res = {i, j, k};
    return res;
}

/**
 * @brief Converts 3D coordinates into a single linear index.
 *
 * This function calculates the linear index corresponding to
 * the (x, y, z) coordinates in a 3D array defined by 'shape'. Each
 * dimension size is stored in shape[0], shape[1], and shape[2], respectively.
 *
 * @param x     The x-coordinate.
 * @param y     The y-coordinate.
 * @param z     The z-coordinate.
 * @param shape A pointer to an array of three integers, containing the
 *              sizes of the dimensions in the order [nx, ny, nz].
 * @return      The computed linear index.
 */
int ravel_index(int x, int y, int z, __global int *shape) {
    int nx = shape[0];  // number of elements in the x dimension
    int ny = shape[1];  // number of elements in the y dimension
    int nz = shape[2];  // number of elements in the z dimension

    // Multiplying x by (y_dim * z_dim) and adding row offsets results in
    // the final single linear index
    return x * ny * nz + y * nz + z;
}