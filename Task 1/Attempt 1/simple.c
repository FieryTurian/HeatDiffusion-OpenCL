#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>
#include <time.h>

#include <CL/cl.h>
#include "simple.h"

typedef struct {
  clarg_type arg_t;
  cl_mem dev_buf;
  double *dhost_buf;
  float *host_buf;
  int    num_elems;
  int    val;
} kernel_arg;

#define MAX_ARG 10


#define die(msg, ...) do {                      \
  (void) fprintf (stderr, msg, ## __VA_ARGS__); \
  (void) fprintf (stderr, "\n");                \
} while (0)

/* global setup */

static cl_platform_id cpPlatform;     /* openCL platform.  */
static cl_device_id device_id;        /* Compute device id.  */
static cl_context context;            /* Compute context.  */
static cl_command_queue commands;     /* Compute command queue.  */
static cl_program program;            /* Compute program.  */
static int num_kernel_args;
static kernel_arg kernel_args[MAX_ARG];

static struct timespec start, stop;
static double kernel_time = 0.0;


cl_int initDevice ( int devType)
{
  cl_int err = CL_SUCCESS;
  cl_uint num_platforms;
  cl_platform_id *cpPlatforms;

  /* Connect to a compute device.  */
  err = clGetPlatformIDs (0, NULL, &num_platforms);
  if (CL_SUCCESS != err) {
    die ("Error: Failed to find a platform!");
  } else {
    cpPlatforms = (cl_platform_id *)malloc( sizeof( cl_platform_id)*num_platforms);
    err = clGetPlatformIDs(num_platforms, cpPlatforms, NULL);

    for(uint i=0; i<num_platforms; i++){
        err = clGetDeviceIDs(cpPlatforms[i], devType, 1, &device_id, NULL);
        if (err == CL_SUCCESS ) {
           cpPlatform = cpPlatforms[i];
           break;
        }
    }
    if (CL_SUCCESS != err) {
      die ("Error: Failed to create a device group!");
    } else { 
      /* Create a compute context.  */
      context = clCreateContext (0, 1, &device_id, NULL, NULL, &err);
      if (!context || err != CL_SUCCESS) {
        die ("Error: Failed to create a compute context!");
      } else {
        /* Create a command commands.  */
        commands = clCreateCommandQueue (context, device_id, 0, &err);
        if (!commands || err != CL_SUCCESS) {
          die ("Error: Failed to create a command commands!");
        }
      }
    }
  }

 return err;
}

cl_int initCPU ()
{
  return initDevice( CL_DEVICE_TYPE_CPU);
}

cl_int initGPU ()
{
  return initDevice( CL_DEVICE_TYPE_GPU);
}

size_t maxWorkItems( int dim)
{
   cl_int err = CL_SUCCESS;
   size_t maxWI = 0;
   size_t max[3];

   if( dim >= 0 && dim < 3) {
      err = clGetDeviceInfo(device_id,
                            CL_DEVICE_MAX_WORK_ITEM_SIZES,
                            3*sizeof(size_t),
                            &max,
                            NULL);
      if (CL_SUCCESS != err) {
         die ("Error: Failed to get device info on work item sizes!");
      } else {
         maxWI = max[dim];
      }
   } else {
      die ("Error: maxWorkItems called with illegal parameter!");
   }

  return maxWI;
}

cl_mem allocDev( size_t n)
{
   cl_int err = CL_SUCCESS;
   cl_mem mem;

   mem = clCreateBuffer (context, CL_MEM_READ_WRITE, n, NULL, &err);
   if( err != CL_SUCCESS) {
      die ("Error %d", err);
      die ("Error: Failed to allocate device memory!");
   }

   return mem;
}

void host2devDoubleArr( double *a, cl_mem ad, size_t n)
{
   cl_int err = CL_SUCCESS;

   err = clEnqueueWriteBuffer( commands, ad, CL_TRUE, 0,
                               sizeof (double) * n,
                               a, 0, NULL, NULL);
   if( CL_SUCCESS != err) {
      die ("Error: Failed to transfer from host to device!");
   }
}

void host2devFloatArr( float *a, cl_mem ad, size_t n)
{
   cl_int err = CL_SUCCESS;

   err = clEnqueueWriteBuffer( commands, ad, CL_TRUE, 0,
                               sizeof (float) * n,
                               a, 0, NULL, NULL);
   if( CL_SUCCESS != err) {
      die ("Error: Failed to transfer from host to device!");
   }
}

void dev2hostDoubleArr( cl_mem ad, double *a, size_t n)
{
   cl_int err = CL_SUCCESS;

   err = clEnqueueReadBuffer (commands, ad, CL_TRUE, 0,
                              sizeof (double) * n,
                              a, 0, NULL, NULL);
   if( CL_SUCCESS != err) {
      die ("Error: Failed to transfer from device to host!");
   }
}

void dev2hostFloatArr( cl_mem ad, float *a, size_t n)
{
   cl_int err = CL_SUCCESS;

   err = clEnqueueReadBuffer (commands, ad, CL_TRUE, 0,
                              sizeof (float) * n,
                              a, 0, NULL, NULL);
   if( CL_SUCCESS != err) {
      die ("Error: Failed to transfer from device to host!");
   }
}

void dev2hostBoolArr( cl_mem ad, bool *a, size_t n)
{
   cl_int err = CL_SUCCESS;

   err = clEnqueueReadBuffer (commands, ad, CL_TRUE, 0,
                              sizeof (bool) * n,
                              a, 0, NULL, NULL);
   if( CL_SUCCESS != err) {
      die ("Error: Failed to transfer from device to host!");
   }
}

cl_kernel createKernel( const char *kernel_source, char *kernel_name)
{
  cl_kernel kernel = NULL;
  cl_int err = CL_SUCCESS;

  /* Create the compute program from the source buffer.  */
  program = clCreateProgramWithSource (context, 1,
                                       (const char **) &kernel_source,
                                       NULL, &err);
  if (!program || err != CL_SUCCESS) {
    die ("Error: Failed to create compute program!");
  }

  /* Build the program executable.  */
  err = clBuildProgram (program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      size_t len;
      char buffer[2048];

      clGetProgramBuildInfo (program, device_id, CL_PROGRAM_BUILD_LOG,
                             sizeof (buffer), buffer, &len);
      die ("Error: Failed to build program executable!\n%s", buffer);
    }

  /* Create the compute kernel in the program.  */
  kernel = clCreateKernel (program, kernel_name, &err);
  if (!kernel || err != CL_SUCCESS) {
    die ("Error: Failed to create compute kernel!");
    kernel = NULL;
  }
  return kernel;
}

cl_kernel setupKernel( const char *kernel_source, char *kernel_name, int num_args, ...)
{
   cl_kernel kernel = NULL;
   cl_int err = CL_SUCCESS;
   va_list ap;
   int i;

   kernel = createKernel( kernel_source, kernel_name);
   num_kernel_args = num_args;
   va_start(ap, num_args);
   for(i=0; (i<num_args) && (kernel != NULL); i++) {
      kernel_args[i].arg_t =va_arg(ap, clarg_type);
      switch( kernel_args[i].arg_t) {
        case DoubleArr:
          kernel_args[i].num_elems = va_arg(ap, int);
          kernel_args[i].dhost_buf = va_arg(ap, double *);
          kernel_args[i].dev_buf = allocDev ( sizeof (double) * kernel_args[i].num_elems);
          host2devDoubleArr ( kernel_args[i].dhost_buf, kernel_args[i].dev_buf, kernel_args[i].num_elems);
          err = clSetKernelArg (kernel, i, sizeof (cl_mem), &kernel_args[i].dev_buf);
          if( CL_SUCCESS != err) {
            die ("Error: Failed to set kernel arg %d!", i);
            kernel = NULL;
          }
          break;
        case FloatArr:
          kernel_args[i].num_elems = va_arg(ap, int);
          kernel_args[i].host_buf = va_arg(ap, float *);
          kernel_args[i].dev_buf = allocDev ( sizeof (float) * kernel_args[i].num_elems);
          host2devFloatArr ( kernel_args[i].host_buf, kernel_args[i].dev_buf, kernel_args[i].num_elems);
          err = clSetKernelArg (kernel, i, sizeof (cl_mem), &kernel_args[i].dev_buf);
          if( CL_SUCCESS != err) {
            die ("Error: Failed to set kernel arg %d!", i);
            kernel = NULL;
          }
          break;
        case IntConst:
          kernel_args[i].val = va_arg(ap, unsigned int);
          err = clSetKernelArg (kernel, i, sizeof (unsigned int), &kernel_args[i].val);
          if( CL_SUCCESS != err) {
            die ("Error: Failed to set kernel arg %d!", i);
            kernel = NULL;
          }
          break;
        default:
          die ("Error: illegal argument tag for executeKernel!");
          kernel = NULL;
      }
   }
   va_end(ap);

   return kernel;
}

cl_int launchKernel( cl_kernel kernel, int dim, size_t *global, size_t *local)
{
  cl_int err;

  clock_gettime( CLOCK_REALTIME, &start);
  if (CL_SUCCESS
      != clEnqueueNDRangeKernel (commands, kernel,
                                 dim, NULL, global, local, 0, NULL, NULL))
    die ("Error: Failed to execute kernel!");

  /* Wait for all commands to complete.  */
  err = clFinish (commands);
  clock_gettime( CLOCK_REALTIME, &stop);
  kernel_time += (stop.tv_sec -start.tv_sec)*1000.0
                  + (stop.tv_nsec -start.tv_nsec)/1000000.0;

  return err;
}

cl_int runKernel( cl_kernel kernel, int dim, size_t *global, size_t *local)
{
  cl_int err = CL_SUCCESS;

  launchKernel( kernel, dim, global, local);

  for( int i=0; i< num_kernel_args; i++) {
    if( kernel_args[i].arg_t == DoubleArr) {
      dev2hostDoubleArr ( kernel_args[i].dev_buf, kernel_args[i].dhost_buf, kernel_args[i].num_elems);
    } else if( kernel_args[i].arg_t == FloatArr) {
      dev2hostFloatArr ( kernel_args[i].dev_buf, kernel_args[i].host_buf, kernel_args[i].num_elems);
    }
  }

  return err;
}

void printKernelTime()
{
  int min, sec;
  double msec;

  min = (int)kernel_time/60000;
  sec = (int)(kernel_time - (min*60000)) / 1000;
  msec = kernel_time - (min*60000) - (sec*1000);

  if (kernel_time > 60000) {
    printf( "total time spent in kernel executions: %d min %d sec %f msec\n", min, sec, msec);
  } else if (kernel_time >1000) {
    printf( "total time spent in kernel executions: %d sec %f msec\n", sec, msec);
  } else {
    printf( "total time spent in kernel executions: %f msec\n", msec);
  }
}

cl_int release() { 
  cl_int err = CL_SUCCESS;

  for( int i=0; i< num_kernel_args; i++) {
    if( (kernel_args[i].arg_t == FloatArr) 
         || (kernel_args[i].arg_t == DoubleArr))
      err = clReleaseMemObject (kernel_args[i].dev_buf);
  }
  
  return err;
}

cl_int freeDevice()
{
  cl_int err;

  err = clReleaseProgram (program);
  err = clReleaseCommandQueue (commands);
  err = clReleaseContext (context);

  return err;
}

void clPrintDevInfo() {
   char device_string[1024];
   clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
   printf("\nCL_DEVICE_NAME: \t\t\t%s\n", device_string);
   
   size_t workgroup_size;
   clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workgroup_size), &workgroup_size, NULL);
   printf("CL_DEVICE_MAX_WORK_GROUP_SIZE: \t\t%lu\n", workgroup_size);
   
   size_t workitem_size[3];
   clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
   printf("CL_DEVICE_MAX_WORK_ITEM_SIZES\t\t%lu / %lu / %lu\n\n", workitem_size[0], workitem_size[1], workitem_size[2]);
}



