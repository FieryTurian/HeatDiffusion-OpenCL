#ifndef SIMPLE_H_
#define SIMPLE_H_

/*******************************************************************************
 *
 * initGPU : sets up the openCL environment for using a GPU.
 *           Note that the system may have more than one GPU in which case
 *           the one that has been pre-configured will be chosen.
 *           If anything goes wrong in the course, error messages will be 
 *           printed to stderr and the last error encountered will be returned.
 *
 ******************************************************************************/
extern cl_int initGPU ();

/*******************************************************************************
 *
 * initCPU : sets up the openCL environment for using the host machine.
 *           If anything goes wrong in the course, error messages will be 
 *           printed to stderr and the last error encountered will be returned.
 *           Note that this may go wrong as not all openCL implementations
 *           support this!
 *
 ******************************************************************************/
extern cl_int initCPU ();

/*******************************************************************************
 *
 * maxWorkItems : returns the maximum number of work items per work group of the
 *                selected device in dimension dim. It requires dim to be
 *                in {0,1,2}.
 *
 ******************************************************************************/
extern size_t maxWorkItems (int dim);

/*******************************************************************************
 *
 * allocDev : returns an openCL device memory identifier for device memory 
 *            of "n" bytes.
 *
 ******************************************************************************/
extern cl_mem allocDev( size_t n);

/*******************************************************************************
 *
 * host2devDoubleArr : transfers "n" elements of the double array "a" on the host
 *                     to the device buffer at "ad".
 *
 ******************************************************************************/
extern void host2devDoubleArr( double *a, cl_mem ad, size_t n);

/*******************************************************************************
 *
 * host2devDoubleArr : transfers "n" elements of the float array "a" on the host
 *                     to the device buffer at "ad".
 *
 ******************************************************************************/
extern void host2devFloatArr( float *a, cl_mem ad, size_t n);

/*******************************************************************************
 *
 * dev2hostDoubleArr : transfers "n" elements of the double array "ad" on the
 *                     device to the host buffer at "a".
 *
 ******************************************************************************/
extern void dev2hostDoubleArr( cl_mem ad, double *a, size_t n);

/*******************************************************************************
 *
 * dev2hostFloatArr : transfers "n" elements of the float array "ad" on the
 *                     device to the host buffer at "a".
 *
 ******************************************************************************/
extern void dev2hostDoubleArr( cl_mem ad, double *a, size_t n);

/*******************************************************************************
 *
 * dev2hostFloatArr : transfers "n" elements of the bool array "ad" on the
 *                     device to the host buffer at "a".
 *
 ******************************************************************************/
extern void dev2hostBoolArr( cl_mem ad, bool *a, size_t n);


/*******************************************************************************
 *
 * createKernel : this routine creates a kernel from the source as string.
 *                It takes the following arguments:
 *               - the kernel source as a string
 *               - the name of the kernel function as string
 *
 ******************************************************************************/
extern cl_kernel createKernel( const char *kernel_source, char *kernel_name);

/*******************************************************************************
 *
 * setupKernel : this routine prepares a kernel for execution. It takes the
 *               following arguments:
 *               - the kernel source as a string
 *               - the name of the kernel function as string
 *               - the number of arguments (must match those specified in the 
 *                 kernel source!)
 *               - followed by the actual arguments. Each argument to the kernel
 *                 results in two or three arguments to this function, depending
 *                 on whether these are pointers to float-arrays or integer values:
 *
 * legal argument sets are:
 *    doubleArr::clarg_type, num_elems::int, pointer::double *,     and
 *    FloatArr::clarg_type, num_elems::int, pointer::float *,     and
 *    IntConst::clarg_type, number::int
 *
 *               If anything goes wrong in the course, error messages will be 
 *               printed to stderr. The pointer to the fully prepared kernel
 *               will be returned.
 *
 *               Note that this function actually performs quite a few openCL
 *               tasks. It compiles the source, it allocates memory on the 
 *               device and it copies over all float arrays. If a more
 *               sophisticated behaviour is needed you may have to fall back to
 *               using openCL directly.
 *
 ******************************************************************************/

typedef enum {
  DoubleArr,
  FloatArr,
  IntConst
} clarg_type;

typedef struct {
    cl_kernel kernel1;
    cl_kernel kernel2;
} kernel_struct;

extern kernel_struct setupKernel( const char *kernel_source, char *kernel_name, int num_args, ...);

/*******************************************************************************
 *
 * launchKernel : this routine executes the kernel given as first argument.
 *             The thread-space is defined through the next two arguments:
 *             <dim> identifies the dimensionality of the thread-space and
 *             <globals> is a vector of length <dim> that gives the upper
 *             bounds for all axes. The argument <local> specifies the size
 *             of the individual warps which need to have the same dimensionality
 *             as the overall range.
 *             If anything goes wrong in the course, error messages will be
 *             printed to stderr and the last error encountered will be returned.
 *
 ******************************************************************************/

extern cl_int launchKernel( cl_kernel kernel, int dim, size_t *global, size_t *local);

/*******************************************************************************
 *
 * runKernel : this routine is similar to launchKernel.
 *             However, in addition to launching the kernel, it also copies back
 *             *all* arguments set up by the previous call to setupKernel!
 *
 ******************************************************************************/

extern cl_int runKernel( cl_kernel kernel, int dim, size_t *global, size_t *local);

/*******************************************************************************
 *
 * printKernelTime : we internally measure the wallclock time that elapses
 *                   during the kernel execution on the device. This routine 
 *                   prints the findings to stdout.
 *                   Note that the measurement does not include any data 
 *                   transfer times for arguments or results! Note also, that
 *                   the only functions that influence the time values are
 *                   launchKernel and runKernel. It does not matter how much
 *                   time elapses between the last call to runKernel and the
 *                   call to printKernelTime!
 *
 ******************************************************************************/

extern void printKernelTime();

/*******************************************************************************
 *
 * freeDevice : this routine releases all acquired ressources.
 *             If anything goes wrong in the course, error messages will be
 *             printed to stderr and the last error encountered will be returned.
 *
 ******************************************************************************/
 
extern cl_int freeDevice();

/*******************************************************************************
 *
 * clPrintDevInfo() : print CL_DEVICE_NAME
 *                    print CL_DEVICE_MAX_WORK_GROUP_SIZE
 *                    print CL_DEVICE_MAX_WORK_ITEM_SIZES
 *
 ******************************************************************************/
 
extern void clPrintDevInfo(); 

#endif /* SIMPLE_H_ */
