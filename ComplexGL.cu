#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <complex.h>
#include <cufft.h>


// Kernel used for assigning values to matrix.


__global__ void assign(int N, cufftDoubleComplex* a, cufftDoubleComplex* a_copy){
  if (blockIdx.x < N && blockIdx.y < N){
    a_copy[blockIdx.x+blockIdx.y*gridDim.x].x = a[blockIdx.x+blockIdx.y*gridDim.x].x;
    a_copy[blockIdx.x+blockIdx.y*gridDim.x].y = a[blockIdx.x+blockIdx.y*gridDim.x].y;
  }
}

// FFT shift kernel, works in GPU

__global__ void fftshift(int N, cufftDoubleComplex* a, cufftDoubleComplex* a_shift){
  if (blockIdx.x < N/2 && blockIdx.y < N/2){
    a_shift[blockIdx.x+blockIdx.y*gridDim.x].x = a[(blockIdx.x+N/2)+(blockIdx.y+N/2)*gridDim.x].x;
    a_shift[blockIdx.x+blockIdx.y*gridDim.x].y = a[(blockIdx.x+N/2)+(blockIdx.y+N/2)*gridDim.x].y;
  }
  else if (blockIdx.x < N/2 && blockIdx.y >= N/2)
    a_shift[blockIdx.x+blockIdx.y*gridDim.x] = a[(blockIdx.x+N/2)+(blockIdx.y-N/2)*gridDim.x];
  else if (blockIdx.x >= N/2 && blockIdx.y < N/2)
    a_shift[blockIdx.x+blockIdx.y*gridDim.x] = a[(blockIdx.x-N/2)+(blockIdx.y+N/2)*gridDim.x];
  else if (blockIdx.x >= N/2 && blockIdx.y >= N/2)
    a_shift[blockIdx.x+blockIdx.y*gridDim.x] = a[(blockIdx.x-N/2)+(blockIdx.y-N/2)*gridDim.x];
}

// Kernel used to compute 2nd derivative

__global__ void del2A(int N, cufftDoubleComplex* del2A_input, cufftDoubleComplex* del2A_output){
  if (blockIdx.x < N && blockIdx.y < N){
    del2A_output[blockIdx.x+blockIdx.y*gridDim.x].x =
            (-((double)(blockIdx.x) + (double)(-N/2))*((double)(blockIdx.x) + (double)(-N/2))
             -((double)(blockIdx.y) + (double)(-N/2))*((double)(blockIdx.y) + (double)(-N/2)))
                    *del2A_input[blockIdx.x+blockIdx.y*gridDim.x].x/N/N;
    del2A_output[blockIdx.x+blockIdx.y*gridDim.x].y =
            (-((double)(blockIdx.x) + (double)(-N/2))*((double)(blockIdx.x) + (double)(-N/2))
             -((double)(blockIdx.y) + (double)(-N/2))*((double)(blockIdx.y) + (double)(-N/2)))
                    *del2A_input[blockIdx.x+blockIdx.y*gridDim.x].y/(N*N);
  }
}

// Kernel used to update A, A1, A2 matrices using pointers to each matrix, with div used to decrement in time.

__global__ void update(int N, double dt, double c1, double c3, int div, double L, cufftDoubleComplex* A, cufftDoubleComplex* d2A, cufftDoubleComplex* A_new){
  if (blockIdx.x < N && blockIdx.y < N){
    A_new[blockIdx.x+blockIdx.y*gridDim.x].x = A[blockIdx.x+blockIdx.y*gridDim.x].x + dt/div*(A[blockIdx.x+blockIdx.y*gridDim.x].x
      + (d2A[blockIdx.x+blockIdx.y*gridDim.x].x - c1*d2A[blockIdx.x+blockIdx.y*gridDim.x].y)*(2*M_PI/L)*(2*M_PI/L)
      - (A[blockIdx.x+blockIdx.y*gridDim.x].x + c3*A[blockIdx.x+blockIdx.y*gridDim.x].y)
      *(A[blockIdx.x+blockIdx.y*gridDim.x].x*A[blockIdx.x+blockIdx.y*gridDim.x].x + A[blockIdx.x+blockIdx.y*gridDim.x].y*A[blockIdx.x+blockIdx.y*gridDim.x].y));

    A_new[blockIdx.x+blockIdx.y*gridDim.x].y = A[blockIdx.x+blockIdx.y*gridDim.x].y + dt/div*(A[blockIdx.x+blockIdx.y*gridDim.x].y
      + (d2A[blockIdx.x+blockIdx.y*gridDim.x].y + c1*d2A[blockIdx.x+blockIdx.y*gridDim.x].x)*(2*M_PI/L)*(2*M_PI/L)
      - (A[blockIdx.x+blockIdx.y*gridDim.x].y - c3*A[blockIdx.x+blockIdx.y*gridDim.x].x)
      *(A[blockIdx.x+blockIdx.y*gridDim.x].x*A[blockIdx.x+blockIdx.y*gridDim.x].x + A[blockIdx.x+blockIdx.y*gridDim.x].y*A[blockIdx.x+blockIdx.y*gridDim.x].y));
  }
}

int main(int argc, char* argv[]){

  // Start runtime clock

  clock_t start_time = clock();

  // Set up args to be taken as inputs

  ptrdiff_t N = atoi(argv[1]);
  double c1 = atof(argv[2]);
  double c3 = atof(argv[3]);
  int M = atoi(argv[4]);

  // Insure same seed is used for all processors

  long int seed = (long int)time(NULL);
  if (argc >= 6){
    seed = atol(argv[5]);
  }
  srand48(seed);

  // Define parameter values

  double L = 128*M_PI;
  int T = 10000;
  double dt = (double)T/M;
  int interval = N/10;

  // Allocate memory within CPU

  cufftDoubleComplex *sol = (cufftDoubleComplex*)malloc(N*N*sizeof(cufftDoubleComplex));

  // Allocate memory within GPU

  cufftDoubleComplex* A;
  cudaMalloc((void**)&A, sizeof(cufftDoubleComplex)*N*N);

  cufftDoubleComplex* A_temp;
  cudaMalloc((void**)&A_temp, sizeof(cufftDoubleComplex)*N*N);

  cufftDoubleComplex* del2A_output;
  cudaMalloc((void**)&del2A_output, sizeof(cufftDoubleComplex)*N*N);

  cufftDoubleComplex* del2A_temp;
  cudaMalloc((void**)&del2A_temp, sizeof(cufftDoubleComplex)*N*N);

  cufftDoubleComplex* del2A_shift;
  cudaMalloc((void**)&del2A_shift, sizeof(cufftDoubleComplex)*N*N);

  // Create and open target file

  FILE *fileid = fopen("ComplexGL.out", "w");

  // ICs

  for (int i = 0; i < N; ++i){
    for (int j = 0; j < N; ++j){
      sol[i*N + j].y = (6*drand48() - 3)/2;
      sol[i*N + j].y = (6*drand48() - 3)/2;
    }
  }
  // Create copy on GPU

  cudaMemcpy(A, sol, N*N*sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

  dim3 meshDim(N,N);
  cufftHandle plan;
  cufftPlan2d(&plan, N, N, CUFFT_Z2Z);

  // start loop

  for (int n = 0; n < M; ++n){

    // Begin 1st

    // Take in 2nd derivative, assign values

    assign<<<meshDim,1>>>(N, A, del2A_output);

    // 2nd derivative

    cufftExecZ2Z(plan, del2A_output, del2A_temp, CUFFT_FORWARD);
    fftshift<<<meshDim,1>>>(N, del2A_temp, del2A_shift);
    del2A<<<meshDim,1>>>(N, del2A_shift, del2A_output);
    fftshift<<<meshDim,1>>>(N, del2A_output, del2A_shift);
    cufftExecZ2Z(plan, del2A_shift, del2A_output, CUFFT_INVERSE);

    // A1 update step, dt/4 step

    update<<<meshDim,1>>>(N, dt, c1, c3, 4, L, A, del2A_output, A_temp);

    // End 1st

    // Begin 2nd

    assign<<<meshDim,1>>>(N, A_temp, del2A_output);

    // 2nd derivative

    cufftExecZ2Z(plan, del2A_output, del2A_temp, CUFFT_FORWARD);
    fftshift<<<meshDim,1>>>(N, del2A_temp, del2A_shift);
    del2A<<<meshDim,1>>>(N, del2A_shift, del2A_output);
    fftshift<<<meshDim,1>>>(N, del2A_output, del2A_shift);
    cufftExecZ2Z(plan, del2A_shift, del2A_output, CUFFT_INVERSE);

    // Update A2, dt/3 step

    update<<<meshDim,1>>>(N, dt, c1, c3, 3, L, A_temp, del2A_output, A);

    // End 2nd

    // Begin 3rd


    assign<<<meshDim,1>>>(N, A, del2A_output);


    cufftExecZ2Z(plan, del2A_output, del2A_temp, CUFFT_FORWARD);
    fftshift<<<meshDim,1>>>(N, del2A_temp, del2A_shift);
    del2A<<<meshDim,1>>>(N, del2A_shift, del2A_output);
    fftshift<<<meshDim,1>>>(N, del2A_output, del2A_shift);
    cufftExecZ2Z(plan, del2A_shift, del2A_output, CUFFT_INVERSE);

    // Update A1, dt/2 step

    update<<<meshDim,1>>>(N, dt, c1, c3, 2, L, A, del2A_output, A_temp);

    // End 3rd


    // Begin 4th

    assign<<<meshDim,1>>>(N, A_temp, del2A_output);

    cufftExecZ2Z(plan, del2A_output, del2A_temp, CUFFT_FORWARD);
    fftshift<<<meshDim,1>>>(N, del2A_temp, del2A_shift);
    del2A<<<meshDim,1>>>(N, del2A_shift, del2A_output);
    fftshift<<<meshDim,1>>>(N, del2A_output, del2A_shift);
    cufftExecZ2Z(plan, del2A_shift, del2A_output, CUFFT_INVERSE);

    // update A, dt/1 step

    update<<<meshDim,1>>>(N, dt, c1, c3, 1, L, A_temp, del2A_output, A);

    // Save final solution and store to CPU

    if ((n+1)%interval == 0){
      cudaThreadSynchronize();
      cudaMemcpy(sol, A, N*N*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
      for (int i = 0; i < N; ++i){
        for (int j = 0; j < N; ++j){
          fwrite(&(sol[i*N+j].x), sizeof(double), 1, fileid);
        }
      }
    }
  }

  // End loop

  // Copy back to CPU

  cudaThreadSynchronize();
  cudaMemcpy(sol, A, N*N*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);

  // Free up all memory used

  free(sol);
  cudaFree(A);
  cudaFree(A_temp);
  cudaFree(del2A_output);
  cudaFree(del2A_temp);
  cudaFree(del2A_shift);
  cufftDestroy(plan);

  // Stop clock, output run time

  clock_t end_time = clock();
  printf("Runtime:%g s.\n", (float)(end_time - start_time)/CLOCKS_PER_SEC);

  return 0;
}
