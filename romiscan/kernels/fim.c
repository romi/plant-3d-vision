#define LOCAL_SIZE 4
#define LOCAL_BUFFER_LENGTH LOCAL_SIZE *LOCAL_SIZE *LOCAL_SIZE
#define OFFSET(i, j, k) i *LOCAL_SIZE *LOCAL_SIZE + j *LOCAL_SIZE + k;

// Point status
#define INACTIVE 0
#define ACTIVE 1
#define CONVERGED 2

#define INF FLT_MAX
#include "common.h"


float solve_quadratic(float a, float b, float c, float F) {
    float temp;
    if (a < b) {
        temp = a;
        a = b;
        b = temp;
    }
    if (b < c) {
        temp = b;
        b = c;
        c = temp;
    }

    float u = INF;
    if (c < INF && F < INF) {
        u = c + F;
        if (u > b) {
            u = 0.5f * (b + c + sqrt(2 * F * F - (b - c) * (b - c)));
            if (u > a) {
                u = (a + b + c) / 3.f +
                    sqrt(4*(a + b + c)*(a + b + c) -
                         12 * (a * a + b * b + c * c - F * F)) /
                        6.f;
            }
        }
    }
    return u;
}

void __kernel update(__global float *sol, __global int *shape,
                     __global float *spd, __global int *active_pts,
                     __global int *point_status, int n_active, int tol, __global int* has_converged) {
    int idx = get_global_id(0);
    if (idx >= n_active)
        return;

    int point_flat_idx = active_pts[idx];
    if (point_status[point_flat_idx] == CONVERGED)
        return;

    int3 point_idx = unravel_index(point_flat_idx, shape);

    float val = sol[point_flat_idx];
    float a, b, c;
    float F;
    if (spd[point_flat_idx] > 0) {
        F = 1.f / spd[point_flat_idx];
    }
    else {
        atomic_xchg(&point_status[point_flat_idx], CONVERGED);
        return;
    }

    a = INF;
    if (point_idx.x != 0) {
        a = min(a, sol[ravel_index(point_idx.x - 1, point_idx.y, point_idx.z, shape)]);
    }
    if (point_idx.x != shape[0] - 1) {
        a = min(a, sol[ravel_index(point_idx.x + 1, point_idx.y, point_idx.z, shape)]);
    } 

    b = INF;
    if (point_idx.y != 0) {
        b = min(b, sol[ravel_index(point_idx.x, point_idx.y - 1, point_idx.z, shape)]);
    }
    if (point_idx.y != shape[1] - 1) {
        b = min(b, sol[ravel_index(point_idx.x, point_idx.y + 1, point_idx.z, shape)]);
    } 

    c = INF;
    if (point_idx.z != 0) {
        c = min(c, sol[ravel_index(point_idx.x, point_idx.y, point_idx.z - 1, shape)]);
    }
    if (point_idx.z != shape[2] - 1) {
        c = min(c, sol[ravel_index(point_idx.x, point_idx.y, point_idx.z + 1, shape)]);
    } 


    float new_val = solve_quadratic(a, b, c, F);
    if (new_val != INF) {
        atomic_xchg(&sol[point_flat_idx], new_val);
        if (val - new_val < tol) {
            atomic_xchg(&point_status[point_flat_idx], CONVERGED);
            return;
        }
    }

    atomic_and(has_converged, 0);
}

void __kernel prune_list(int __global *active_pts, int __global *active_pts_out,
                         __global int *point_status, int n_active,
                         int __global *cnt) {
    int idx = get_global_id(0);
    if (idx >= n_active)
        return;
    if (point_status[active_pts[idx]] == ACTIVE) {
        int old = atomic_add(cnt, 1);
        active_pts_out[old] = active_pts[idx];
    }
}

void try_and_add(int __global *active_pts, __global int *point_status, int idx,
                 int __global *cnt) {
    if (point_status[idx] == INACTIVE) {
        int old = atomic_add(cnt, 1);
        active_pts[old] = idx;
        point_status[idx] = ACTIVE;
    }
}

void __kernel add_neighbours(int __global *active_pts, int __global *shape,
                             __global int *point_status, int n_active,
                             int __global *cnt) {

    int idx = get_global_id(0);
    if (idx >= n_active)
        return;
    int point_flat_idx = active_pts[idx];
    int new_point_flat_idx;
    int3 point_idx = unravel_index(point_flat_idx, shape);

    if (point_idx.x != 0) {
        new_point_flat_idx =
            ravel_index(point_idx.x - 1, point_idx.y, point_idx.z, shape);
        try_and_add(active_pts, point_status, new_point_flat_idx, cnt);
    }

    if (point_idx.x != shape[0] - 1) {
        new_point_flat_idx =
            ravel_index(point_idx.x + 1, point_idx.y, point_idx.z, shape);
        try_and_add(active_pts, point_status, new_point_flat_idx, cnt);
    }

    if (point_idx.y != 0) {
        new_point_flat_idx =
            ravel_index(point_idx.x, point_idx.y - 1, point_idx.z, shape);
        try_and_add(active_pts, point_status, new_point_flat_idx, cnt);
    }

    if (point_idx.y != shape[1] - 1) {
        new_point_flat_idx =
            ravel_index(point_idx.x, point_idx.y + 1, point_idx.z, shape);
        try_and_add(active_pts, point_status, new_point_flat_idx, cnt);
    }

    if (point_idx.z != 0) {
        new_point_flat_idx =
            ravel_index(point_idx.x, point_idx.y, point_idx.z - 1, shape);
        try_and_add(active_pts, point_status, new_point_flat_idx, cnt);
    }

    if (point_idx.z != shape[2] - 1) {
        new_point_flat_idx =
            ravel_index(point_idx.x, point_idx.y, point_idx.z + 1, shape);
        try_and_add(active_pts, point_status, new_point_flat_idx, cnt);
    }
}

