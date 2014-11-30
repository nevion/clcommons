#ifndef CLCOMMONS_WORK_GROUP_H
#define CLCOMMONS_WORK_GROUP_H

#ifndef WG_SIZE_MAX
#define WG_SIZE_MAX 256
#endif

//power of 2 variants are more efficient in memory
#define WORK_GROUP_FUNCTION_MEMORY_SIZE_POWER2_(p2_max_wg_size) (2*p2_max_wg_size)
#define WORK_GROUP_FUNCTION_MEMORY_SIZE_POWER2 WORK_GROUP_FUNCTION_MEMORY_SIZE_POWER2_(WG_SIZE_MAX)
//power of 2 variants are more efficient in memory
#define WORK_GROUP_FUNCTION_MEMORY_SIZE(max_log2down_of_groupsize) (4 * max_log2down_of_groupsize)
#define WORK_GROUP_FUNCTION_MAX_MEMORY_SIZE WORK_GROUP_FUNCTION_MEMORY_SIZE_POWER2

enum clcommon_clcommon_scan_reduce_t{
    CLCOMMON_SCAN = 0,
    CLCOMMON_REDUCE = 1
};

enum clcommon_scan_t{
    CLCOMMON_SCAN_INCLUSIVE = 0,
    CLCOMMON_SCAN_EXCLUSIVE = 1
};

#define clcommon_add(a, b) ((a) + (b))
#define clcommon_min(a, b) ((a) <= (b) ? (a) : (b))
#define clcommon_max(a, b) ((a) >= (b) ? (a) : (b))

#define WORK_GROUP_SCAN_IMPL_POWER2(T, op, identity, scan_or_reduce, exclusive)                                         \
    /* Set first half of local memory to zero to make room for scanning */                                              \
    size_t l_id = get_local_linear_id();                                                                          \
    const size_t wg_size = get_workgroup_size();                                                                        \
    /*set identity values*/                                                                                             \
    lmem[l_id] = identity;                                                                                              \
                                                                                                                        \
    l_id += wg_size;                                                                                                    \
    lmem[l_id] = val;                                                                                                   \
    lds_barrier();                                                                                                      \
                                                                                                                        \
    for (int i = 1; i < wg_size; i *= 2){                                                                               \
        T t = lmem[l_id -  i];                                                                                          \
        lds_barrier();                                                                                                  \
        lmem[l_id] = op(lmem[l_id], t);                                                                                 \
        lds_barrier();                                                                                                  \
    }                                                                                                                   \
    if(scan_or_reduce == CLCOMMON_SCAN){                                                                                \
        return lmem[l_id-(exclusive == CLCOMMON_SCAN_EXCLUSIVE)];                                                       \
    }else{                                                                                                              \
        return lmem[wg_size*2 - 1];                                                                                     \
    }

#define MAKE_WORK_GROUP_SCAN_POWER2(T, LDSType, operator, identity)                                                                                         \
T PASTE(clc_work_group_scan_power2_, T)(T val, operator, enum clcommon_scan_reduce_t scan_or_reduce, enum clcommon_scan_t exclusive, __local LDSType *lmem){\
    WORK_GROUP_SCAN_IMPL_POWER2(T, op, identity, , exclusive)                                                                                               \
}

#define WORK_GROUP_SCAN_IMPL_NONPOWER2(T, op, identity, scan_or_reduce, exclusive)                                                    \
    /* Set first half of local memory to zero to make room for scanning */                                                            \
    size_t _l_id = get_local_linear_id();                                                                                             \
    size_t l_id = _l_id;                                                                                                              \
    const size_t wg_size = get_workgroup_size();                                                                                      \
    const size_t log2DownWgSize = log2Down(wg_size);                                                                                  \
    const size_t active_wg_size = (1<<log2DownWgSize);                                                                                \
    const size_t second_pass_offset = 2 * active_wg_size;                                                                             \
    /*const size_t offset = _l_id < active_wg_size ? 0 : second_pass_offset - active_wg_size;*/                                       \
                                                                                                                                      \
    if(_l_id < active_wg_size){                                                                                                       \
        lmem[_l_id] = identity;                                                                                                       \
        lmem[_l_id + second_pass_offset] = identity;                                                                                  \
    }                                                                                                                                 \
    lds_barrier();                                                                                                                    \
                                                                                                                                      \
    l_id += active_wg_size;                                                                                                           \
                                                                                                                                      \
    if(_l_id < active_wg_size){                                                                                                       \
        lmem[l_id] = val;                                                                                                             \
        if(_l_id >= (wg_size - active_wg_size)){                                                                                      \
            lmem[l_id + second_pass_offset] = identity;                                                                               \
        }                                                                                                                             \
    }else{/*_l_id >= active_wg_size*/                                                                                                 \
        lmem[l_id - active_wg_size + second_pass_offset] = val;                                                                       \
    }                                                                                                                                 \
    lds_barrier();                                                                                                                    \
                                                                                                                                      \
    for (int i = 1; i < active_wg_size; i *= 2){                                                                                      \
        T t;                                                                                                                          \
        if(_l_id < active_wg_size){                                                                                                   \
            t = lmem[l_id -  i];                                                                                                      \
        }                                                                                                                             \
        lds_barrier();                                                                                                                \
        if(_l_id < active_wg_size){                                                                                                   \
            lmem[l_id] = op(lmem[l_id], t);                                                                                           \
        }                                                                                                                             \
        lds_barrier();                                                                                                                \
    }                                                                                                                                 \
                                                                                                                                      \
    /*reduce the final first pass reduction with the first input of the second pass*/                                                 \
    if(_l_id == active_wg_size - 1){                                                                                                  \
        T first_half_reduction = lmem[l_id];                                                                                          \
        const size_t second_half_index = second_pass_offset + 0 + active_wg_size;                                                     \
        lmem[second_half_index] = op(lmem[second_half_index], first_half_reduction);                                                  \
    }                                                                                                                                 \
    lds_barrier();                                                                                                                    \
                                                                                                                                      \
    for (int i = 1; i < active_wg_size; i *= 2){                                                                                      \
        T t;                                                                                                                          \
        if(_l_id < active_wg_size){                                                                                                   \
            t = lmem[second_pass_offset + l_id -  i];                                                                                 \
        }                                                                                                                             \
        lds_barrier();                                                                                                                \
        lmem[second_pass_offset + l_id] = op(lmem[second_pass_offset + l_id], t);                                                     \
        lds_barrier();                                                                                                                \
    }                                                                                                                                 \
                                                                                                                                      \
    if(scan_or_reduce == CLCOMMON_SCAN){                                                                                              \
        size_t output_index;                                                                                                          \
        if(exclusive == CLCOMMON_SCAN_EXCLUSIVE){                                                                                     \
            output_index = _l_id < active_wg_size ?                                                                                   \
                l_id - 1:                                                                                                             \
                (_l_id == active_wg_size ? l_id - exclusive : l_id - active_wg_size + second_pass_offset - 1);                        \
            /*output_index = _l_id < active_wg_size ?            */                                                                   \
            /*    l_id - 1:                                      */                                                                   \
            /*    l_id - active_wg_size + second_pass_offset - 1;*/                                                                   \
        }else{                                                                                                                        \
            output_index = _l_id < active_wg_size ?                                                                                   \
                l_id:                                                                                                                 \
                l_id - active_wg_size + second_pass_offset;/*-> (_l_id - active_wg_size) + second_pass_offset + active_wg_size;*/     \
            /*(_l_id - active_wg_size) + second_pass_offset + active_wg_size;*/                                                       \
        }                                                                                                                             \
        return lmem[output_index];                                                                                                    \
    }else{/*reduce*/                                                                                                                  \
        return lmem[second_pass_offset + 2*active_wg_size - 1];                                                                       \
    }

#define MAKE_WORK_GROUP_FUNCTIONS(T, LDSType, TMIN, TMAX)                                                                \
T PASTE(clc_work_group_scan_inclusive_add_, T) (T val, __local LDSType *lmem){                                                     \
    if(isPowerOf2(get_workgroup_size())){                                                                                          \
        WORK_GROUP_SCAN_IMPL_POWER2(T, clcommon_add, 0, CLCOMMON_SCAN, CLCOMMON_SCAN_INCLUSIVE)                                    \
    }else{                                                                                                                         \
        WORK_GROUP_SCAN_IMPL_NONPOWER2(T, clcommon_add, 0, CLCOMMON_SCAN, CLCOMMON_SCAN_INCLUSIVE)                                 \
    }                                                                                                                              \
}                                                                                                                                  \
T PASTE(clc_work_group_scan_exclusive_add_, T) (T val, __local LDSType *lmem){                                                     \
    if(isPowerOf2(get_workgroup_size())){                                                                                          \
        WORK_GROUP_SCAN_IMPL_POWER2(T, clcommon_add, 0, CLCOMMON_SCAN, CLCOMMON_SCAN_EXCLUSIVE)                                    \
    }else{                                                                                                                         \
        WORK_GROUP_SCAN_IMPL_NONPOWER2(T, clcommon_add, 0, CLCOMMON_SCAN, CLCOMMON_SCAN_EXCLUSIVE)                                 \
    }                                                                                                                              \
}                                                                                                                                  \
T PASTE(clc_work_group_reduce_sum_, T) (T val, __local LDSType *lmem){                                                             \
    if(isPowerOf2(get_workgroup_size())){                                                                                          \
        WORK_GROUP_SCAN_IMPL_POWER2(T, clcommon_add, 0, CLCOMMON_REDUCE, CLCOMMON_SCAN_INCLUSIVE)                                  \
    }else{                                                                                                                         \
        WORK_GROUP_SCAN_IMPL_NONPOWER2(T, clcommon_add, 0, CLCOMMON_REDUCE, CLCOMMON_SCAN_INCLUSIVE)                               \
    }                                                                                                                              \
}                                                                                                                                  \
T PASTE(clc_work_group_reduce_min_, T) (T val, __local LDSType *lmem){                                                             \
    if(isPowerOf2(get_workgroup_size())){                                                                                          \
        WORK_GROUP_SCAN_IMPL_POWER2(T, clcommon_min, TMAX, CLCOMMON_REDUCE, CLCOMMON_SCAN_INCLUSIVE)                               \
    }else{                                                                                                                         \
        WORK_GROUP_SCAN_IMPL_NONPOWER2(T, clcommon_min, TMAX, CLCOMMON_REDUCE, CLCOMMON_SCAN_INCLUSIVE)                            \
    }                                                                                                                              \
}                                                                                                                                  \
T PASTE(clc_work_group_reduce_max_, T) (T val, __local LDSType *lmem){                                                             \
    if(isPowerOf2(get_workgroup_size())){                                                                                          \
        WORK_GROUP_SCAN_IMPL_POWER2(T, clcommon_max, TMIN, CLCOMMON_REDUCE, CLCOMMON_SCAN_INCLUSIVE)                               \
    }else{                                                                                                                         \
        WORK_GROUP_SCAN_IMPL_NONPOWER2(T, clcommon_max, TMIN, CLCOMMON_REDUCE, CLCOMMON_SCAN_INCLUSIVE)                            \
    }                                                                                                                              \
}                                                                                                                                  \
T PASTE(clc_work_group_broadcast1_, T)(T val, size_t local_id, __local T *value){                                                  \
    if(local_id == get_local_id(0)){                                                                                               \
        *value = val;                                                                                                              \
    }                                                                                                                              \
    lds_barrier();                                                                                                                 \
    return *value;                                                                                                                 \
}                                                                                                                                  \
T PASTE(clc_work_group_broadcast2_, T)(T val, size_t local_id_0, size_t local_id_1, __local T *value){                             \
    if((local_id_0 == get_local_id(0)) & (local_id_1 == get_local_id(1))){                                                         \
        *value = val;                                                                                                              \
    }                                                                                                                              \
    lds_barrier();                                                                                                                 \
    return *value;                                                                                                                 \
}                                                                                                                                  \
T PASTE(clc_work_group_broadcast3_, T)(T val, size_t local_id_0, size_t local_id_1, size_t local_id_2, __local T *value){          \
    if((local_id_0 == get_local_id(0)) & (local_id_1 == get_local_id(1)) & (local_id_2 == get_local_id(2))){                       \
        *value = val;                                                                                                              \
    }                                                                                                                              \
    lds_barrier();                                                                                                                 \
    return *value;                                                                                                                 \
}

#ifdef ENABLE_CL_CPP

#include "clcommon/util_type.h"

//lmem: 2*wg_size, the upper half is used as a scratchpad
template<typename T, clcommon_scan_t exclusive, typename LDSType = T>
INLINE T clc_work_group_sum_power2(T val, __local T *lmem){
    WORK_GROUP_SCAN_IMPL_POWER2(T, clcommon_add, 0, exclusive);
}

#define IDENTITY_MAX_VALUE NumericTraits<T>::MAX_VALUE()
#define IDENTITY_MIN_VALUE NumericTraits<T>::MIN_VALUE()

template<typename T>
INLINE T clc_work_group_scan_inclusive_add(T val, __local T *lmem){
    if(isPowerOf2(get_workgroup_size())){
        WORK_GROUP_SCAN_IMPL_POWER2(T, clcommon_add, 0, CLCOMMON_SCAN, CLCOMMON_SCAN_INCLUSIVE)
    }else{
        WORK_GROUP_SCAN_IMPL_NONPOWER2(T, clcommon_add, 0, CLCOMMON_SCAN, CLCOMMON_SCAN_INCLUSIVE)
    }
}

template<typename T>
INLINE T clc_work_group_scan_exclusive_add(T val, __local T *lmem){
    if(isPowerOf2(get_workgroup_size())){
        WORK_GROUP_SCAN_IMPL_POWER2(T, clcommon_add, 0, CLCOMMON_SCAN, CLCOMMON_SCAN_EXCLUSIVE)
    }else{
        WORK_GROUP_SCAN_IMPL_NONPOWER2(T, clcommon_add, 0, CLCOMMON_SCAN, CLCOMMON_SCAN_EXCLUSIVE)
    }
}

template<typename T>
INLINE T clc_work_group_reduce_min(T val, __local T *lmem){
    if(isPowerOf2(get_workgroup_size())){
        WORK_GROUP_SCAN_IMPL_POWER2(T, clcommon_min, IDENTITY_MAX_VALUE, CLCOMMON_REDUCE, CLCOMMON_SCAN_INCLUSIVE)
    }else{
        WORK_GROUP_SCAN_IMPL_NONPOWER2(T, clcommon_min, IDENTITY_MAX_VALUE, CLCOMMON_REDUCE, CLCOMMON_SCAN_INCLUSIVE)
    }
}

template<typename T>
T clc_work_group_reduce_max(T val, __local T *lmem){
    if(isPowerOf2(get_workgroup_size())){
        WORK_GROUP_SCAN_IMPL_POWER2(T, clcommon_max, IDENTITY_MIN_VALUE, CLCOMMON_REDUCE, CLCOMMON_SCAN_INCLUSIVE)
    }else{
        WORK_GROUP_SCAN_IMPL_NONPOWER2(T, clcommon_max, IDENTITY_MIN_VALUE, CLCOMMON_REDUCE, CLCOMMON_SCAN_INCLUSIVE)
    }
}

template<typename T>
T clc_work_group_reduce_sum(T val, __local T *lmem){
    if(isPowerOf2(get_workgroup_size())){
        WORK_GROUP_SCAN_IMPL_POWER2(T, clcommon_add, IDENTITY_MIN_VALUE, CLCOMMON_REDUCE, CLCOMMON_SCAN_INCLUSIVE)
    }else{
        WORK_GROUP_SCAN_IMPL_NONPOWER2(T, clcommon_add, IDENTITY_MIN_VALUE, CLCOMMON_REDUCE, CLCOMMON_SCAN_INCLUSIVE)
    }
}

template<typename T>
T clc_work_group_broadcast(T val, size_t local_id, __local T *value){
    if(local_id == get_local_id(0)){
        *value = val;
    }
    lds_barrier();
    return *value;
}

template<typename T>
T clc_work_group_broadcast(T val, size_t local_id_0, size_t local_id_1, __local T *value){
    if((local_id_0 == get_local_id(0)) & (local_id_1 == get_local_id(1))){
        *value = val;
    }
    lds_barrier();
    return *value;
}

template<typename T>
T clc_work_group_broadcast(T val, size_t local_id_0, size_t local_id_1, size_t local_id_2, __local T *value){
    if((local_id_0 == get_local_id(0)) & (local_id_1 == get_local_id(1)) & (local_id_2 == get_local_id(2))){
        *value = val;
    }
    lds_barrier();
    return *value;
}
#undef IDENTITY_MAX_KEY
#undef IDENTITY_MIN_KEY

#endif

#endif
