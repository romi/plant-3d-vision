__kernel void check(__read_only image2d_t mask,
            __global uchar* labels,
            __global float* intrinsics,
            __global float* rot,
            __global float* tvec, __global float* volinfo, __global int* shape) {
const sampler_t samplerA = CLK_NORMALIZED_COORDS_FALSE |
                       CLK_ADDRESS_NONE |
                       CLK_FILTER_NEAREST;

int i = get_global_id(0);
int j = get_global_id(1);
int k = get_global_id(2);

int nx = shape[0];
int ny = shape[1];
int nz = shape[2];

int idx_label = i*ny*nz + j*nz + k;


if (labels[idx_label] == 1)
{
    return;
}

float f_x = intrinsics[0];
float f_y = intrinsics[1];
float c_x = intrinsics[2];
float c_y = intrinsics[3];

float x = volinfo[0] + i*volinfo[3];
float y = volinfo[1] + i*volinfo[3];
float z = volinfo[2] + i*volinfo[3];

float p_x = rot[0] * x +
        rot[1] * y +
        rot[2] * z + tvec[0];

float p_y = rot[3] * x +
        rot[4] * y +
        rot[5] * z + tvec[1];

float p_z = rot[6] * x +
        rot[7] * y +
        rot[8] * z + tvec[2];

if (p_z < 0) {
    return;
}
p_x = p_x/p_z * f_x + c_x;
p_y = p_y/p_z * f_y + c_y;

int p_x_int = (int) p_x;
int p_y_int = (int) p_y;
int2 image_coord = {p_x_int, p_y_int};

if (p_x_int < 0 || p_x_int > get_image_width(mask)-1) {
return;
}
if (p_y_int < 0 || p_y_int > get_image_height(mask)-1) {
return;
}
if (read_imageui(mask, samplerA, image_coord).x == 0) {
    labels[idx_label] = 1;
} else if (labels[i] == 0) { // Mark a voxel the first time it is seen
    labels[idx_label] = 2;
}
}
