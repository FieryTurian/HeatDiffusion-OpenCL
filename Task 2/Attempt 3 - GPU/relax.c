#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>
#include "simple.h"

#define N 10000000   // length of the vectors
#define EPS 0.1      // convergence criterium
#define HEAT 100.0   // heat value on the boundary

struct timespec start, stop;

void printTimeElapsed(char *text)
{
  double elapsed = (stop.tv_sec -start.tv_sec)*1000.0
                  + (double)(stop.tv_nsec -start.tv_nsec)/1000000.0;
  printf( "%s: %f msec\n", text, elapsed);
}

//
// allocate a vector of length "n"
//
double *allocVector(int n)
{
   double *v;
   v = (double *)malloc( n*sizeof(double));
   return v;
}

//
// allocate a bool vector of length "n"
//
bool *allocStable(int n)
{
   bool *s;
   s = (bool *)malloc( n*sizeof(bool));
   return s;
}

//
// initialise the values of the given vector "out" of length "n"
//
void init(double *out, int n)
{
   int i;

   for(i=1; i<n; i++) {
      out[i] = 0;
   }
   out[0] = HEAT;
}

//
// initialise the values of the given vector "out" of length "n"
//
void binit(bool *out, int n)
{
   int i;

   for(i=0; i<n; i++) {
      out[i] = false;
   }
}

//
// print the values of a given vector "out" of length "n"
//
void print(double *out, int n)
{
   int i;

   printf("<");
   for(i=0; i<n; i++) {
      printf(" %f", out[i]);
   }
   printf(">\n");
}

//
//relax function in kernel source
//
const char *KernelSource =                                      "\n"
  "__kernel void relax(                                          \n"
  "   __global double* in,                                       \n"
  "   __global double* out,                                      \n"
  "   __global bool* stable,                                     \n"
  "   const double eps,                                          \n"
  "   const unsigned int count)                                  \n"
  "{                                                             \n"
  "   int i = get_global_id(0);                                  \n"
  "   int n = get_global_size(0);                                \n"
  "   if (i == 0)                                                \n"
  "      stable[0] = true;                                       \n"
  "   if (i > 0 && i < n-1) {                                    \n"
  "      out[i] = 0.25*in[i-1] + 0.5*in[i] + 0.25*in[i+1];       \n"
  "   } else {                                                   \n"
  "      out[i] = in[i];                                         \n"
  "   }                                                          \n"
  "   if (fabs(in[i] - out[i]) > eps)                            \n"
  "      stable[0] = false;                                      \n"
  "}                                                             \n"
  "\n";

int main()
{
   cl_int err;
   kernel_struct kernels;
   size_t global[1];
   size_t local[1];
  
   double *a,*b;
   bool *stable;
   int n, count;
   int iterations = 0;

   a = allocVector(N);
   b = allocVector(N);
   stable = allocStable(1);

   init(a, N);
   init(b, N);
   binit(stable, 1);

   n = N;
   count = 0;
   
   local[0] = 32;
   printf("work group size: %d\n", (int)local[0]);
   global[0] = n;
   printf("global work size: %d\n\n", n);

   printf("size   : %d M (%d MB)\n", n/1000000, (int)(n*sizeof(double) / (1024*1024)));
   printf("heat   : %f\n", HEAT);
   printf("epsilon: %f\n", EPS);
   
   err = initGPU();
   //clPrintDevInfo();
      
   if (err == CL_SUCCESS) {
      clock_gettime(1, &start);
      kernels = setupKernel(KernelSource, "relax", 5, DoubleArr, n, a, DoubleArr, n, b, BoolArr, 1, stable, DoubleConst, EPS, IntConst, n);

      do {         
         if(count == 0) {
            runKernel(kernels.kernel1, 1, global, local);
            count++;
         } else {
            runKernel(kernels.kernel2, 1, global, local);
            count--;
         }
         
         iterations++;
      } while(!stable[0]);
      
      clock_gettime(1, &stop);
      
      printf("Number of iterations: %d\n", iterations);
      printTimeElapsed("GPU time spent");
      printKernelTime();
      
      err = clReleaseKernel(kernels.kernel1);
      err = clReleaseKernel(kernels.kernel2);
      err = freeDevice();
   }

   return 0;
}
