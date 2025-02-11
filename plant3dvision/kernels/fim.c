/**
 * @file fim.c
 * @brief Fast Iterative Method (FIM) core update routines and helper functions.
 */

#include "common.h"   // Required for ravel_index, unravel_index, etc.

#define LOCAL_SIZE 4
#define LOCAL_BUFFER_LENGTH (LOCAL_SIZE * LOCAL_SIZE * LOCAL_SIZE)
#define OFFSET(i, j, k) ((i) * LOCAL_SIZE * LOCAL_SIZE + (j) * LOCAL_SIZE + (k))

/* Point Status Definitions */
#define INACTIVE 0
#define ACTIVE 1
#define CONVERGED 2

#define INF FLT_MAX


/**
 * @brief Solves a piecewise quadratic equation to update a distance value.
 *
 * This function rearranges the inputs (a, b, c) in ascending order,
 * then uses a tried-and-true formula for the “Fast Marching” style update.
 *
 * @param a    The smallest neighbor value encountered (initially).
 * @param b    The middle neighbor value encountered (initially).
 * @param c    The largest neighbor value encountered (initially).
 * @param F    Inverse speed (1 / speed), or INF if speed is zero.
 *
 * @return     The computed updated distance or INF if not solvable.
 */
float solve_quadratic(float a, float b, float c, float F) {
    float temp;

    /* Reorder the inputs so that a <= b <= c (after swapping) */
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

    /* Only proceed if c and F are finite */
    if (c < INF && F < INF) {
        /* Initial update guess: c + F */
        u = c + F;

        /* If our guess is too large, solve a 2D quadratic form */
        if (u > b) {
            u = 0.5f * (b + c + sqrt(2.0f * F * F - (b - c) * (b - c)));

            /* If still too large, solve the 3D version */
            if (u > a) {
                float sum = (a + b + c);
                u = (sum / 3.0f)
                    + sqrt(4.0f * sum * sum
                           - 12.0f * (a * a + b * b + c * c - F * F)) / 6.0f;
            }
        }
    }
    return u;
}

/**
 * @brief Kernel that updates distance values for active points.
 *
 * This kernel reads active points, retrieves their current distance,
 * and updates them by considering their neighbors’ distances.
 * The status of each point is updated if it has converged.
 *
 * @param sol           Global buffer of solution distances.
 * @param shape         Dimensions of the 3D grid (x, y, z).
 * @param spd           Global buffer of speeds (used to compute F).
 * @param active_pts    Global buffer of currently active point indices.
 * @param point_status  Global buffer indicating status of each point.
 * @param n_active      Number of active points.
 * @param tol           Convergence tolerance threshold.
 * @param has_converged A single-integer flag used for global convergence check.
 */
__kernel void update(__global float *sol,
                     __global int *shape,
                     __global float *spd,
                     __global int *active_pts,
                     __global int *point_status,
                     int n_active,
                     int tol,
                     __global int* has_converged) {
    /* Thread ID in the global range */
    int idx = get_global_id(0);
    if (idx >= n_active)
        return;

    /* Retrieve the actual point index from the active list */
    int point_flat_idx = active_pts[idx];

    /* Skip if already converged */
    if (point_status[point_flat_idx] == CONVERGED)
        return;

    /* Compute x, y, z from flat index */
    int3 point_idx = unravel_index(point_flat_idx, shape);

    float val = sol[point_flat_idx];

    /* Speed to 1/speed (F) if valid, else converge immediately */
    float F;
    if (spd[point_flat_idx] > 0) {
        F = 1.0f / spd[point_flat_idx];
    } else {
        atomic_xchg(&point_status[point_flat_idx], CONVERGED);
        return;
    }

    /* Gather the minimum values among neighbors in each dimension */
    float a = INF, b = INF, c = INF;

    /* -- X dimension neighbors -- */
    if (point_idx.x != 0) {
        float tmp = sol[ravel_index(point_idx.x - 1,
                                    point_idx.y,
                                    point_idx.z, shape)];
        a = min(a, tmp);
    }
    if (point_idx.x != (shape[0] - 1)) {
        float tmp = sol[ravel_index(point_idx.x + 1,
                                    point_idx.y,
                                    point_idx.z, shape)];
        a = min(a, tmp);
    }

    /* -- Y dimension neighbors -- */
    if (point_idx.y != 0) {
        float tmp = sol[ravel_index(point_idx.x,
                                    point_idx.y - 1,
                                    point_idx.z, shape)];
        b = min(b, tmp);
    }
    if (point_idx.y != (shape[1] - 1)) {
        float tmp = sol[ravel_index(point_idx.x,
                                    point_idx.y + 1,
                                    point_idx.z, shape)];
        b = min(b, tmp);
    }

    /* -- Z dimension neighbors -- */
    if (point_idx.z != 0) {
        float tmp = sol[ravel_index(point_idx.x,
                                    point_idx.y,
                                    point_idx.z - 1, shape)];
        c = min(c, tmp);
    }
    if (point_idx.z != (shape[2] - 1)) {
        float tmp = sol[ravel_index(point_idx.x,
                                    point_idx.y,
                                    point_idx.z + 1, shape)];
        c = min(c, tmp);
    }

    /* Attempt to solve the local update equation */
    float new_val = solve_quadratic(a, b, c, F);

    /* If a valid update was found, replace solution and check convergence */
    if (new_val != INF) {
        atomic_xchg(&sol[point_flat_idx], new_val);

        /* If the change is less than or equal to tol, mark as converged */
        if ((val - new_val) < tol) {
            atomic_xchg(&point_status[point_flat_idx], CONVERGED);
            return;
        }
    }

    /* Mark that the global solution has not fully converged for this iteration */
    atomic_and(has_converged, 0);
}

/**
 * @brief Kernel that prunes converged points from the active list.
 *
 * Iterates over the current active list (active_pts), checks if each
 * point is still ACTIVE. If so, copies it to the output list.
 *
 * @param active_pts     Current list of active points.
 * @param active_pts_out Output list of still-active points.
 * @param point_status   Global buffer with point statuses.
 * @param n_active       Number of active points in the current list.
 * @param cnt            Counter to track how many points are kept.
 */
__kernel void prune_list(__global int *active_pts,
                         __global int *active_pts_out,
                         __global int *point_status,
                         int n_active,
                         __global int *cnt) {
    int idx = get_global_id(0);
    if (idx >= n_active)
        return;

    /* Keep the point if its status is ACTIVE */
    if (point_status[active_pts[idx]] == ACTIVE) {
        int old = atomic_add(cnt, 1);
        active_pts_out[old] = active_pts[idx];
    }
}

/**
 * @brief Helper function to add a neighboring point to active_pts if it is INACTIVE.
 *
 * If the neighbor’s status is INACTIVE, it becomes ACTIVE and is appended to the list.
 *
 * @param active_pts    Current buffer of active points.
 * @param point_status  Global buffer holding the status of each point.
 * @param idx           The flat index of the neighbor point.
 * @param cnt           Counter used to place a new active point in the correct position.
 */
void try_and_add(__global int *active_pts,
                 __global int *point_status,
                 int idx,
                 __global int *cnt) {
    if (point_status[idx] == INACTIVE) {
        int old = atomic_add(cnt, 1);  // Get the insertion index atomically
        active_pts[old] = idx;
        point_status[idx] = ACTIVE;
    }
}

/**
 * @brief Kernel that checks neighbors of each active point and adds them if needed.
 *
 * Each thread processes one active point to add valid INACTIVE neighbors
 * (in the 6 directions along x, y, z) to the active list.
 *
 * @param active_pts    Global array of current active point indices.
 * @param shape         Dimensions of the 3D grid (x, y, z).
 * @param point_status  Global buffer holding point status.
 * @param n_active      Number of active points.
 * @param cnt           Counter used for adding new active points safely.
 */
void __kernel add_neighbours(int __global *active_pts, int __global *shape,
                             __global int *point_status, int n_active,
                             int __global *cnt) {

    int idx = get_global_id(0);
    if (idx >= n_active)
        return;

    /* Retrieve the point index from the active list */
    int point_flat_idx = active_pts[idx];
    int new_point_flat_idx;
    int3 point_idx = unravel_index(point_flat_idx, shape);

    /* Check neighbor in -X direction */
    if (point_idx.x != 0) {
        new_point_flat_idx =
            ravel_index(point_idx.x - 1, point_idx.y, point_idx.z, shape);
        try_and_add(active_pts, point_status, new_point_flat_idx, cnt);
    }

    /* Check neighbor in +X direction */
    if (point_idx.x != shape[0] - 1) {
        new_point_flat_idx =
            ravel_index(point_idx.x + 1, point_idx.y, point_idx.z, shape);
        try_and_add(active_pts, point_status, new_point_flat_idx, cnt);
    }

    /* Check neighbor in -Y direction */
    if (point_idx.y != 0) {
        new_point_flat_idx =
            ravel_index(point_idx.x, point_idx.y - 1, point_idx.z, shape);
        try_and_add(active_pts, point_status, new_point_flat_idx, cnt);
    }

    /* Check neighbor in +Y direction */
    if (point_idx.y != shape[1] - 1) {
        new_point_flat_idx =
            ravel_index(point_idx.x, point_idx.y + 1, point_idx.z, shape);
        try_and_add(active_pts, point_status, new_point_flat_idx, cnt);
    }

    /* Check neighbor in -Z direction */
    if (point_idx.z != 0) {
        new_point_flat_idx =
            ravel_index(point_idx.x, point_idx.y, point_idx.z - 1, shape);
        try_and_add(active_pts, point_status, new_point_flat_idx, cnt);
    }

    /* Check neighbor in +Z direction */
    if (point_idx.z != shape[2] - 1) {
        new_point_flat_idx =
            ravel_index(point_idx.x, point_idx.y, point_idx.z + 1, shape);
        try_and_add(active_pts, point_status, new_point_flat_idx, cnt);
    }
}

