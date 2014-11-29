#pragma OPENCL EXTENSION cl_amd_printf : enable

#include "clcommons/common.h"
#include "clcommons/util_type.h"
#include "clcommons/work_group.h"

typedef ITEMT ItemT;
typedef LDSITEMT LDSItemT;

MAKE_WORK_GROUP_FUNCTIONS(ITEMT, LDSITEMT, TMIN, TMAX)

//global dims: 1 work_dims: wg_size
__kernel void
wg_inclusive_sum(__global const ItemT * restrict input_image_p, __global ItemT * restrict scan_p){
    const LDSItemT v = input_image_p[get_local_id(0)];
    __local LDSItemT lmem[WORK_GROUP_FUNCTION_MAX_MEMORY_SIZE];
    LDSItemT i = PASTE(clc_work_group_scan_inclusive_add_, ITEMT)(v, lmem);
    scan_p[get_local_id(0)] = i;
}

__kernel void
wg_exclusive_sum(__global const ItemT * restrict input_image_p, __global ItemT * restrict scan_p){
    const LDSItemT v = input_image_p[get_local_id(0)];
    __local LDSItemT lmem[WORK_GROUP_FUNCTION_MAX_MEMORY_SIZE];
    LDSItemT i = PASTE(clc_work_group_scan_exclusive_add_, ITEMT)(v, lmem);
    scan_p[get_local_id(0)] = i;
}

__kernel void
wg_reduce_sum(__global const ItemT * restrict input_image_p, __global ItemT * restrict reduction){
    const LDSItemT v = input_image_p[get_local_id(0)];
    __local LDSItemT lmem[WORK_GROUP_FUNCTION_MAX_MEMORY_SIZE];

    LDSItemT i = PASTE(clc_work_group_reduce_sum_, ITEMT)(v, lmem);
    if(get_local_id(0) == get_local_size(0) - 1){
        *reduction = i;
    }
}

__kernel void
wg_reduce_min(__global const ItemT * restrict input_image_p, __global ItemT * restrict reduction){
    const LDSItemT v = input_image_p[get_local_id(0)];
    __local LDSItemT lmem[WORK_GROUP_FUNCTION_MAX_MEMORY_SIZE];

    LDSItemT i = PASTE(clc_work_group_reduce_min_, ITEMT)(v, lmem);
    if(get_local_id(0) == get_local_size(0) - 1){
        *reduction = i;
    }
}

__kernel void
wg_reduce_max(__global const ItemT * restrict input_image_p, __global ItemT * restrict reduction){
    const LDSItemT v = input_image_p[get_local_id(0)];
    __local LDSItemT lmem[WORK_GROUP_FUNCTION_MAX_MEMORY_SIZE];

    LDSItemT i = PASTE(clc_work_group_reduce_max_, ITEMT)(v, lmem);
    if(get_local_id(0) == get_local_size(0) - 1){
        *reduction = i;
    }
}
