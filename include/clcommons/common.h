#ifndef CLCOMMONS_COMMON_H_
#define CLCOMMONS_COMMON_H_

#ifdef DEBUG
#define INLINE
#define assert(x) \
        if (! (x)) \
        { \
            printf((__constant char *)"Assert(%s) failed in %s:%d\n", #x, __FILE__, __LINE__); \
        }
#define assert_val(x, val) \
        if (! (x)) \
        { \
            printf((__constant char *)"Assert(%s) failed in %s:%d:  %d\n", #x, __FILE__, __LINE__, val); \
        }
#else
    #define INLINE inline
    #define assert(X)
    #define assert_val(X, val)
    //#define printf(fmt, ...)
#endif

//don't take this near upper limits of integral type
INLINE uint divUp(const uint x, const uint divisor){
    return (x + (divisor - 1)) / divisor;
}

INLINE uint divUpSafe(const uint x, const uint divisor){
    const uint k = x / divisor;
    return k * divisor >= x ? k : k + 1;
}

INLINE uint roundUpToMultiple(const uint x, const uint multiple){
    return divUp(x, multiple) * multiple;
}

INLINE uint log2Down(uint x){
    uint power = 1;
    x >>= 1;
    uint i = 0;
    while(x){
        power <<= 1;
        x >>= 1;
        i++;
    }
    return i;
}

INLINE uint log2Up(uint x){
    uint power = log2Down(x);
    return (1<<power) < x ? power + 1 : power;
}

INLINE uint isPowerOf2(uint x){
    return (x & (x - 1)) == 0;
}

#define STRINGIFY2( x) #x
#define STRINGIFY(x) STRINGIFY2(x)

#define PASTE2( a, b) a##b
#define PASTE( a, b) PASTE2( a, b)

uint get_workgroup_size(){
    return get_local_size(0) * get_local_size(1) * get_local_size(2);
}

#if __OPENCL_VERSION__ < 200
#define work_group_barrier barrier
uint get_local_linear_id(){
    return get_local_id(0) + get_local_id(1) * get_local_size(0) + get_local_id(2) * get_local_size(0) * get_local_size(1);
}
#endif

#define ensure_lds_barrier() work_group_barrier(CLK_LOCAL_MEM_FENCE)

#if defined(AMD_GPU_ARCH) || defined(NVIDIA_ARCH)
    #ifdef PROMISE_WG_IS_WAVEFRONT
        #define lds_barrier() assert(get_workgroup_size() <= DEVICE_WAVEFRONT_SIZE); mem_fence(CLK_LOCAL_MEM_FENCE);
    #else
        #define lds_barrier() if(get_workgroup_size() >= DEVICE_WAVEFRONT_SIZE){ work_group_barrier(CLK_LOCAL_MEM_FENCE); }else{ mem_fence(CLK_LOCAL_MEM_FENCE); }
    #endif
#else
#define lds_barrier() ensure_lds_barrier()
#endif

#endif /* CLCOMMONS_COMMON_H_ */
