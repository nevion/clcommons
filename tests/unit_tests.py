import unittest

from .common import *
from .platform import *

def prefix_sum_test(input_data, kernel, dtype, exclusive):
    #print 'wg_size: %r'%(wg_size,)
    wg_size = len(input_data)

    ref_result = prefix_sum(input_data, exclusive, dtype=dtype)

    src_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, wg_size*dtype.itemsize)
    dst_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, wg_size*dtype.itemsize)
    cl_result = np.zeros(wg_size, dtype = dtype)
    cl.enqueue_copy(queue, src_buf, input_data, is_blocking = True)

    event = kernel(queue, (1,), (wg_size,), src_buf, dst_buf, g_times_l=True)
    event.wait()
    cl.enqueue_copy(queue, cl_result, dst_buf, is_blocking = True)
    diffs = np.where(cl_result != ref_result)[0]
    if len(diffs) > 0:
        print 'reference:'
        print ref_result
        print 'cl_result:'
        print cl_result
        from IPython import embed; embed()
        #print 'wg_size %d different starting at %d, span of %d'%(wg_size, diffs[0], len(tdata) - diffs[0] - 1)
        return (wg_size, diffs[0], len(cl_result) - diffs[0])
    else:
        return (wg_size, 0, 0)

def reduce_test(input_data, kernel, reduction_fun, dtype):
    #print 'wg_size: %r'%(wg_size,)
    wg_size = len(input_data)

    ref_result = None
    if reduction_fun is np.sum:
        ref_result = reduction_fun(input_data, dtype=dtype)
    else:
        ref_result = reduction_fun(input_data)
    src_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, wg_size*dtype.itemsize)
    dst_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, wg_size*dtype.itemsize)
    cl_result = np.zeros(1, dtype = dtype)
    cl.enqueue_copy(queue, src_buf, input_data, is_blocking = True)
    event = kernel(queue, (1,), (wg_size,), src_buf, dst_buf, g_times_l=True)
    event.wait()
    cl.enqueue_copy(queue, cl_result, dst_buf, is_blocking = True)
    return (wg_size, ref_result != cl_result, ref_result, cl_result)

def load_tests(loader, tests, pattern):
    class WorkGroupTestCases(unittest.TestCase):
        def __init__(self, method, dtype, program, debug = False):
            self.dtype = dtype
            self.debug = debug
            self.program = program
            self.method = method
            unittest.TestCase.__init__(self, method)

        def gen_input_random(self, wg_size):
            np.random.seed(24)
            input_data = np.array([np.random.randint(0, 1+1) for x in range(wg_size)], self.dtype)
            return input_data
        def gen_input_zeros(self, wg_size):
            return np.zeros(wg_size, self.dtype)
        def gen_input_ones(self, wg_size):
            return np.ones(wg_size, self.dtype)
        def gen_reduction_input(self, wg_size, reduction_fun):
            dtype = self.dtype
            np.random.seed(24)
            denom = (1<<(dtype.itemsize*8 - 1))
            min_value, max_value = np.iinfo(dtype).min/denom, (np.iinfo(dtype).max+1)/denom
            input_data = None
            if reduction_fun is np.sum and dtype==np.uint8:
                input_data = np.array([np.random.randint(0, 2) for x in range(wg_size)], dtype)
            if reduction_fun is np.sum and dtype==np.int8:
                input_data = np.array([np.random.randint(-2, 3) for x in range(wg_size)], dtype)
            else:
                input_data = np.array([np.random.randint(min_value, max_value) for x in range(wg_size)], dtype)
            return input_data
        def test_inclusive_prefix_sum(self):
            kernel = self.program.wg_inclusive_sum
            sum_tests_rand = np.array([prefix_sum_test(self.gen_input_random(x), kernel, self.dtype, 0) for x in range(1, 257)], self.dtype)
            self.assertTrue(np.all(sum_tests_rand[:, 2] == 0))
            sum_tests_zeros = np.array([prefix_sum_test(self.gen_input_zeros(x), kernel, self.dtype, 0) for x in range(1, 257)], self.dtype)
            self.assertTrue(np.all(sum_tests_zeros[:, 2] == 0))
            sum_tests_ones = np.array([prefix_sum_test(self.gen_input_ones(x), kernel, self.dtype, 0) for x in range(1, 257)], self.dtype)
            self.assertTrue(np.all(sum_tests_ones[:, 2] == 0))
        def test_exclusive_prefix_sum(self):
            kernel = self.program.wg_exclusive_sum
            sum_tests_rand = np.array([prefix_sum_test(self.gen_input_random(x), kernel, self.dtype, 1) for x in range(1, 257)], self.dtype)
            self.assertTrue(np.all(sum_tests_rand[:, 2] == 0))
            sum_tests_zeros = np.array([prefix_sum_test(self.gen_input_zeros(x), kernel, self.dtype, 1) for x in range(1, 257)], self.dtype)
            self.assertTrue(np.all(sum_tests_zeros[:, 2] == 0))
            sum_tests_ones = np.array([prefix_sum_test(self.gen_input_ones(x), kernel, self.dtype, 1) for x in range(1, 257)], self.dtype)
            self.assertTrue(np.all(sum_tests_ones[:, 2] == 0))
        def test_reduce_min(self):
            kernel = self.program.wg_reduce_min
            reduce_tests_rand = np.array([reduce_test(self.gen_reduction_input(x, np.min), kernel, np.min, self.dtype) for x in range(1, 257)], self.dtype)
            self.assertTrue(np.all(reduce_tests_rand[:, 1] == 0))
            reduce_tests_zeros = np.array([reduce_test(self.gen_input_zeros(x), kernel, np.min, self.dtype) for x in range(1, 257)], self.dtype)
            self.assertTrue(np.all(reduce_tests_zeros[:, 1] == 0))
            reduce_tests_ones = np.array([reduce_test(self.gen_input_ones(x), kernel, np.min, self.dtype) for x in range(1, 257)], self.dtype)
            self.assertTrue(np.all(reduce_tests_ones[:, 1] == 0))
        def test_reduce_max(self):
            kernel = self.program.wg_reduce_max
            reduce_tests_rand = np.array([reduce_test(self.gen_reduction_input(x, np.max), kernel, np.max, self.dtype) for x in range(1, 257)], self.dtype)
            self.assertTrue(np.all(reduce_tests_rand[:, 1] == 0))
            reduce_tests_zeros = np.array([reduce_test(self.gen_input_zeros(x), kernel, np.max, self.dtype) for x in range(1, 257)], self.dtype)
            self.assertTrue(np.all(reduce_tests_zeros[:, 1] == 0))
            reduce_tests_ones = np.array([reduce_test(self.gen_input_ones(x), kernel, np.max, self.dtype) for x in range(1, 257)], self.dtype)
            self.assertTrue(np.all(reduce_tests_ones[:, 1] == 0))
        def test_reduce_sum(self):
            kernel = self.program.wg_reduce_sum
            reduce_tests_rand = np.array([reduce_test(self.gen_input_random(x), kernel, np.sum, self.dtype) for x in range(1, 257)], self.dtype)
            self.assertTrue(np.all(reduce_tests_rand[:, 1] == 0))
            reduce_tests_zeros = np.array([reduce_test(self.gen_input_zeros(x), kernel, np.sum, self.dtype) for x in range(1, 257)], self.dtype)
            self.assertTrue(np.all(reduce_tests_zeros[:, 1] == 0))
            reduce_tests_ones = np.array([reduce_test(self.gen_input_ones(x), kernel, np.sum, self.dtype) for x in range(1, 257)], self.dtype)
            self.assertTrue(np.all(reduce_tests_ones[:, 1] == 0))
        def __str__(self):
            return 'WorkGroup %r dtype: %r'%(self.method, self.dtype)

    suite = unittest.TestSuite()
    dtypes = [np.uint8, np.int8, np.int16, np.uint16, np.int32, np.uint32]
    methods = ['test_inclusive_prefix_sum', 'test_exclusive_prefix_sum', 'test_reduce_min', 'test_reduce_max', 'test_reduce_sum']
    debug = True
    for dtype in map(np.dtype, dtypes):
        ItemT = type_mapper(dtype)
        LDSItemT = 'uint'
        if ItemT[0] != 'u':
            LDSItemT = 'int'
        iinfo = np.iinfo(dtype)
        KERNEL_FLAGS = "-D ITEMT={ItemT} -D LDSITEMT={LDSItemT} -D TMIN={TMIN} -D TMAX={TMAX}".format(ItemT=ItemT, LDSItemT=LDSItemT, TMIN=iinfo.min, TMAX=iinfo.max)
        CL_FLAGS = "-I %s -cl-std=CL1.2 %s" %(common_lib_path, KERNEL_FLAGS)
        CL_FLAGS = cl_opt_decorate(debug, CL_FLAGS)
        print '%r compile flags: %s'%(ItemT, CL_FLAGS)

        CL_SOURCE = file(os.path.join(base_path, 'test_work_group_kernels.cl'), 'r').read()
        program = cl.Program(ctx, CL_SOURCE).build(options=CL_FLAGS)
        for method in methods:
            suite.addTest(WorkGroupTestCases(method, dtype, program))
    return suite
