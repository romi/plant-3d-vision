/**
 * geodesic - OpenCL kernel to iteratively trace points through a 3D vector field
 *
 * This kernel uses the gradient components (gx, gy, gz) and an intensity field
 * (values) to guide each point along geodesic paths. Each point is updated in
 * one iteration based on the gradient direction scaled by step_size, moving
 * the point to a new position. If this movement leads to a non-improving or
 * invalid position (e.g., out of bounds or failing a threshold check), the
 * corresponding label is cleared. Otherwise, the kernel updates a vote counter
 * for the newly occupied voxel, indicating that the point has contributed to
 * that location.
 *
 * Parameters:
 *   gx, gy, gz      - 3D image objects holding the x, y, and z components of
 *                     the gradient field.
 *   values          - 3D image object with image intensity or scalar values
 *                     used to evaluate if a step is valid.
 *   votes           - Global array for counting how many points converge to
 *                     each voxel (accumulated by atomic operations).
 *   points          - (x, y, z) coordinates of each point; updated in place.
 *   labels          - Per-point flags to indicate if a point is still valid
 *                     for further tracing.
 *   points_remain   - Single-integer flag (atomic) indicating whether any
 *                     point is still valid after an iteration.
 *   shape           - Dimensions of the 3D volume: [nx, ny, nz].
 *   step_size       - Scale factor for each gradient step taken by the points.
 */

__kernel void geodesic(__read_only image3d_t gx,
                       __read_only image3d_t gy,
                       __read_only image3d_t gz,
                       __read_only image3d_t values,
                       __global int* votes,
                       __global float* points,
                       __global uchar* labels,
                       __global int* points_remain,
                       __global int* shape,
                       float step_size)
{
    // Specify how sampling from 3D images is done (no address mode and linear filtering)
    const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_NONE |
                            CLK_FILTER_LINEAR;

    // Clear the flag indicating that any points remain valid for tracing;
    // a single pass sets it to 1 if a valid point is encountered.
    atomic_and(points_remain, 0);

    // Obtain global ID: each thread processes one point
    int i = get_global_id(0);

    // Load volume dimensions
    int nx = shape[0];
    int ny = shape[1];
    int nz = shape[2];

    // If the point's label is 0, it's not traced
    if (!labels[i])
    {
        return;
    }

    // Current point coordinates (x, y, z)
    float x = points[3*i];
    float y = points[3*i+1];
    float z = points[3*i+2];

    // Convert to float4 in the order (z, y, x, 0) for reading from 3D images
    float4 pt = (float4)(z, y, x, 0.0f);

    // Read the intensity (val) and gradient components from respective images
    float val = read_imagef(values, smp, pt).x;
    float gx_val = read_imagef(gx, smp, pt).x;
    float gy_val = read_imagef(gy, smp, pt).x;
    float gz_val = read_imagef(gz, smp, pt).x;

    // Move the point backwards along the gradient by step_size
    x -= gx_val * step_size;
    y -= gy_val * step_size;
    z -= gz_val * step_size;

    // Construct the new float4 for reading the new intensity value
    float4 new_pt = (float4)(z, y, x, 0.0f);
    float new_val = read_imagef(values, smp, new_pt).x;

    // Update the point coordinates in global memory
    points[3*i]   = x;
    points[3*i+1] = y;
    points[3*i+2] = z;

    // Check if the movement is both improving (lower intensity) and above a threshold
    // If not, discard this point by clearing its label
    if (new_val < step_size || new_val >= val) {
        labels[i] = 0;
        return;
    }

    // If the point is still valid, mark that at least one point remains
    atomic_or(points_remain, 1);

    // Cast the new position to integer indices
    int xi = (int)x;
    int yi = (int)y;
    int zi = (int)z;

    // Ensure the indices are within volume bounds;
    // If they are, increment the vote counter for that voxel
    if(xi >= 0 && xi < nx &&
       yi >= 0 && yi < ny &&
       zi >= 0 && zi < nz)
    {
        int idx = xi * ny * nz + yi * nz + zi;
        atomic_add(&votes[idx], 1);
    }

    return;
}
