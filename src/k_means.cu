#include "k_means.h"
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cmath>

int N = 10000000;
int K = 4;
int SAMPLES_PER_THREAD = 1;
int THREADS_PER_BLOCK = 128;
int N_BLOCKS = ceil((float)N / (float)THREADS_PER_BLOCK / (float)SAMPLES_PER_THREAD);

// Coordinates of samples (on host)
float *h_sample_x;
float *h_sample_y;

// Size of clusters (on host)
int *h_cluster_size;

__device__ float dist(float a_x, float a_y, float b_x, float b_y)
{
  return (b_x - a_x) * (b_x - a_x) + (b_y - a_y) * (b_y - a_y);
}

void init(float *d_sample_x, float *d_sample_y, int *d_cluster_indices,
          float *d_centroids_x, float *d_centroids_y)
{
  // Initialize randomization seed
  srand(10);

  // Create host arrays for generation on host
  h_sample_x = (float *)malloc(N * sizeof(float));
  h_sample_y = (float *)malloc(N * sizeof(float));
  h_cluster_size = (int *)malloc(K * sizeof(int));

  // Generate samples and initialize cluster indices with dummy value
  for (int i = 0; i < N; i++)
  {
    h_sample_x[i] = (float)rand() / RAND_MAX;
    h_sample_y[i] = (float)rand() / RAND_MAX;
  }

  int n_bytes = 4 * N;
  int k_bytes = 4 * K;

  // Copy data to device
  cudaMemcpy(d_sample_x, h_sample_x, n_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sample_y, h_sample_y, n_bytes, cudaMemcpyHostToDevice);
  cudaMemset(d_cluster_indices, -1, n_bytes);
  cudaMemcpy(d_centroids_x, d_sample_x, k_bytes, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_centroids_y, d_sample_y, k_bytes, cudaMemcpyDeviceToDevice);
  checkCUDAError("Init error");
}

void calc_centroids(float *d_centroids_x, float *d_centroids_y,
                    float *d_centroid_sum_x, float *d_centroid_sum_y, 
                    int *d_cluster_count)
{
    float h_centroids_x[K];
    float h_centroids_y[K];
    float h_centroid_sum_x[K], h_centroid_sum_y[K];
    int h_cluster_count[K];
    cudaMemcpy(h_centroid_sum_x, d_centroid_sum_x, K * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_centroid_sum_y, d_centroid_sum_y, K * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cluster_count, d_cluster_count, K * 4, cudaMemcpyDeviceToHost);

    for (int i = 0; i < K; i++) {
      h_centroids_x[i] = h_centroid_sum_x[i] / h_cluster_count[i];
      h_centroids_y[i] = h_centroid_sum_y[i] / h_cluster_count[i];
      //printf("%d- %f, %f, %d\n", i, h_centroid_sum_x[i], h_centroid_sum_y[i], h_cluster_count[i]);
    }
    //printf("\n");

    cudaMemcpy(d_centroids_x, h_centroids_x, K * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids_y, h_centroids_y, K * 4, cudaMemcpyHostToDevice);
}

// Used if SAMPLES_PER_THREAD > 1
__global__
void distribute_elements_kernel(float *d_sample_x, float *d_sample_y,
                                float *d_centroids_x, float *d_centroids_y,
                                int *d_cluster_indices,
                                int *d_cluster_count, float *d_centroids_sum_x,
                                float *d_centroids_sum_y,
                                bool *changed, int N, const int K, const int SperT)
{
  int bid = threadIdx.x;
  int id = blockDim.x * blockIdx.x + bid;

  // Shared memory array divided as follows:
  // {sample_0.x, sample0.y, sample1.x, ..., centroid_0.x, centroid_0.y, ...,
  //  new_centroid_0_x_sum, new_centroid_0_y_sum, cluster_0_count, ...}
  // The second line of data is for new centroid calculation
  extern __shared__ float shared_samples[];
  int shared_ind = bid * SperT * 2; //index of x of first sample for this thread
  int cent_start = blockDim.x * SperT * 2; //index of x coord of centroid 0
  int new_start = cent_start + K * 2; //index of x coord of new centroid 0 sum
  
  for (int i = 0; i < SperT && id * SperT + i < N; i++) {
    shared_samples[shared_ind + i * 2] = d_sample_x[id * SperT + i];
    shared_samples[shared_ind + i * 2 + 1] = d_sample_y[id * SperT + i];
  }
  
  if (bid < K) {
    shared_samples[cent_start + bid * 2] = d_centroids_x[bid];
    shared_samples[cent_start + bid * 2 + 1] = d_centroids_y[bid];
    shared_samples[new_start + bid * 3] = 0;
    shared_samples[new_start + bid * 3 + 1] = 0;
    shared_samples[new_start + bid * 3 + 2] = 0;
  }
  
  __syncthreads();

  for (int i = 0; i < SperT && id * SperT + i < N; i++)
  {
    int cluster_index = 0;
    float min = dist(shared_samples[shared_ind + i * 2], shared_samples[shared_ind + i * 2 + 1],
                     shared_samples[cent_start], shared_samples[cent_start + 1]);
    //float min = dist(d_sample_x[id + i], d_sample_y[id + i],
    //                 d_centroids_x[0], d_centroids_y[0]);
    for (int j = 1; j < K; j++)
    {
      float d = dist(shared_samples[shared_ind + i * 2], shared_samples[shared_ind + i * 2 + 1],
                     shared_samples[cent_start + j * 2], shared_samples[cent_start + j * 2 + 1]);
      //float d = dist(d_sample_x[id + i], d_sample_y[id + i],
      //               d_centroids_x[j], d_centroids_y[j]);
      if (d < min)
      {
        cluster_index = j;
        min = d;
      }
    }

    // Update data
    if (cluster_index != d_cluster_indices[id * SperT + i])
    {
      *changed = true;
    }
    d_cluster_indices[id * SperT + i] = cluster_index;
    atomicAdd_block(&shared_samples[new_start + cluster_index * 3], shared_samples[shared_ind + i * 2]);
    atomicAdd_block(&shared_samples[new_start + cluster_index * 3 + 1], shared_samples[shared_ind + i * 2 + 1]);
    atomicAdd_block(&shared_samples[new_start + cluster_index * 3 + 2], 1);
  }

  __syncthreads();

  if (bid < K) {
    atomicAdd(&d_centroids_sum_x[bid], shared_samples[new_start + bid * 3]);
    atomicAdd(&d_centroids_sum_y[bid], shared_samples[new_start + bid * 3 + 1]);
    atomicAdd(&d_cluster_count[bid], shared_samples[new_start + bid * 3 + 2]);
  }
}

int main(int argc, char **argv)
{
  // # Argument parsing
  if (argc > 1)
  {
    sscanf(argv[1], "%d", &N);
    if (argc > 2)
    {
      sscanf(argv[2], "%d", &K);
      if (argc > 3)
      {
        sscanf(argv[3], "%d", &THREADS_PER_BLOCK);
        if (argc > 4)
        {
          sscanf(argv[4], "%d", &SAMPLES_PER_THREAD);
        }
      }
    }
  }
  N_BLOCKS = ceil((float)N / (float)THREADS_PER_BLOCK / (float)SAMPLES_PER_THREAD);

  // # Allocate memory on device
  float *d_sample_x, *d_sample_y, *d_centroids_x, *d_centroids_y;
  float *d_centroid_sum_x, *d_centroid_sum_y;
  int *d_cluster_indices, *d_cluster_count;

  int n_bytes = 4 * N;
  int k_bytes = 4 * K;

  cudaMalloc((void **)&d_sample_x, n_bytes);
  cudaMalloc((void **)&d_sample_y, n_bytes);
  cudaMalloc((void **)&d_cluster_indices, n_bytes);
  cudaMalloc((void **)&d_centroids_x, k_bytes);
  cudaMalloc((void **)&d_centroids_y, k_bytes);
  cudaMalloc((void **)&d_centroid_sum_x, k_bytes);
  cudaMalloc((void **)&d_centroid_sum_y, k_bytes);
  cudaMalloc((void **)&d_cluster_count, k_bytes);
  cudaMemset(d_centroid_sum_x, 0, k_bytes);
  cudaMemset(d_centroid_sum_y, 0, k_bytes);
  cudaMemset(d_cluster_count, 0, k_bytes);
  checkCUDAError("Malloc error");

  // # Generate values and place in device memory
  init(d_sample_x, d_sample_y, d_cluster_indices, d_centroids_x, d_centroids_y);

  // # Begin main loop
  // ## Prepare convergence flag in host/device
  bool h_changed, *d_changed;
  cudaMalloc((void **)&d_changed, sizeof(bool));
  cudaMemset(d_changed, 0, sizeof(bool));
  // ## First iteration sample distribution
  // ### sh_bytes is the amount of bytes of shared memory. Shared memory will 
  // ### have all the block's respective samples, all the current centroid
  // ### coordinates, and room for new centroid calculation at the end.
  int sh_bytes = SAMPLES_PER_THREAD * THREADS_PER_BLOCK * sizeof(float) * 2 + K * 5 * sizeof(float);
  distribute_elements_kernel<<<N_BLOCKS, THREADS_PER_BLOCK, sh_bytes>>>(d_sample_x,
                                                                        d_sample_y,
                                                                        d_centroids_x,
                                                                        d_centroids_y,
                                                                        d_cluster_indices,
                                                                        d_cluster_count,
                                                                        d_centroid_sum_x,
                                                                        d_centroid_sum_y,
                                                                        d_changed,
                                                                        N, K, SAMPLES_PER_THREAD);
  cudaDeviceSynchronize();
  checkCUDAError("Device sync");
  // ## Get convergence flag from device
  cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
  checkCUDAError("Getting flag");
  int iterations;
  for (iterations = 0; iterations < 21 && h_changed; iterations++)
  {
    // ## Recalculate centroids
    calc_centroids(d_centroids_x, d_centroids_y, d_centroid_sum_x, d_centroid_sum_y, d_cluster_count);

    // ## Reset 'changed' flags on device and host
    h_changed = false;
    cudaMemset(d_changed, 0, sizeof(bool));

    // ## Reset accumulator arrays
    cudaMemset(d_centroid_sum_x, 0, k_bytes);
    cudaMemset(d_centroid_sum_y, 0, k_bytes);
    cudaMemset(d_cluster_count, 0, k_bytes);

    // ## Redistribute elements
    distribute_elements_kernel<<<N_BLOCKS, THREADS_PER_BLOCK, sh_bytes>>>(d_sample_x,
                                                                          d_sample_y,
                                                                          d_centroids_x,
                                                                          d_centroids_y,
                                                                          d_cluster_indices,
                                                                          d_cluster_count,
                                                                          d_centroid_sum_x,
                                                                          d_centroid_sum_y,
                                                                          d_changed,
                                                                          N, K, SAMPLES_PER_THREAD);
    cudaDeviceSynchronize();                                                                  
    // ## Get 'changed' flag from device
    cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
  }

  // # Print results
  printf("N = %d, K = %d\n", N, K);
  float h_centroids_x[K], h_centroids_y[K];
  int h_cluster_count[K];
  cudaMemcpy(h_centroids_x, d_centroids_x, K * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_centroids_y, d_centroids_y, K * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_cluster_count, d_cluster_count, K * 4, cudaMemcpyDeviceToHost);
  for (int i = 0; i < K; i++)
  {
    printf("Center: (%1.3f, %1.3f) : Size: %d\n", h_centroids_x[i],
           h_centroids_y[i], h_cluster_count[i]);
  }
  printf("Iterations: %d\n", iterations);
}
