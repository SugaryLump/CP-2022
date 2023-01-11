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

void calc_centroids(int *d_cluster_indices,
                    float *d_centroids_x, float *d_centroids_y)
{
  // New host array of cluster indices because they´ve been calculated
  // on device
  int *h_cluster_indices = (int*)malloc(sizeof(int) * N);
  // Grab new cluster indices from device
  cudaMemcpy(h_cluster_indices, d_cluster_indices, N * 4, cudaMemcpyDeviceToHost);
  // New host array for centroid  coordinates because we need to calculate
  // it on host before sending it over to device
  float h_centroids_x[K], h_centroids_y[K];
  // Local cluster coordinates sum
  float cluster_x[K], cluster_y[K];
  // Local array for parallelization of middle loop with openmp (reduce needs
  // local array)
  int p_cluster_size[K];

  // Init cluster sum/size arrays
#pragma omp smd
  for (int i = 0; i < K; i++)
  {
    cluster_x[i] = 0;
    cluster_y[i] = 0;
    p_cluster_size[i] = 0;
  }

  // Calculate cluster sums/size with openmp parallelization
#pragma omp parallel for reduction(+ : cluster_x, cluster_y, p_cluster_size)
  for (int i = 0; i < N; i++)
  {
    cluster_x[h_cluster_indices[i]] += h_sample_x[i];
    cluster_y[h_cluster_indices[i]] += h_sample_y[i];
    p_cluster_size[h_cluster_indices[i]]++;
  }

  // Calculate centroids
  for (int i = 0; i < K; i++)
  {
    h_cluster_size[i] = p_cluster_size[i];
    h_centroids_x[i] = cluster_x[i] / p_cluster_size[i];
    h_centroids_y[i] = cluster_y[i] / p_cluster_size[i];
    // printf("%d- size: %d, centroid %f, %f\n", i, h_cluster_size[i], h_centroids_x[i], h_centroids_y[i]);
  }
  // printf("\n");

  // Send new centroids to device
  cudaMemcpy(d_centroids_x, h_centroids_x, K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_centroids_y, h_centroids_y, K * sizeof(float), cudaMemcpyHostToDevice);
  checkCUDAError("Centroids copy error");

  free(h_cluster_indices);
}

// Used if SAMPLES_PER_THREAD > 1
__global__
void distribute_elements_kernel(float *d_sample_x, float *d_sample_y,
                                float *d_centroids_x, float *d_centroids_y,
                                int *d_cluster_indices,
                                bool *changed, int N, const int K, const int SperT)
{
  int bid = threadIdx.x;
  int id = blockDim.x * blockIdx.x + bid;

  //Shared memory array that divided as follows:
  // {sample_0.x, sample0.y, sample1.x, etc..., centroid_0.x, centroid_0.y, etc...}
  //extern __shared__ float shared_samples[];
  //int cent_start = blockDim.x * SperT; //index of x coord of centroid 0
  //
  //for (int i = bid; i - bid < SperT &&  id + i - bid < N; i++) {
  //  shared_samples[i * 2] = d_sample_x[id + i - bid];
  //  shared_samples[i * 2 + 1] = d_sample_y[id + i - bid];
  //}
  //
  //if (bid < K) {
  //  shared_samples[cent_start + bid * 2] = d_centroids_x[bid];
  //  shared_samples[cent_start + bid * 2 + 1] = d_centroids_y[bid];
  //}
  //
  //__syncthreads();

  int start_i = id * SperT;
  for (int i = id * SperT; i - start_i < SperT && i < N; i++)
  {
    // Find nearest cluster
    int cluster_index = 0;

    float min = dist(d_sample_x[i], d_sample_y[i],
                     d_centroids_x[0], d_centroids_y[0]);
    for (int j = 1; j < K; j++)
    {
      float d = dist(d_sample_x[i], d_sample_y[i],
                     d_centroids_x[j], d_centroids_y[j]);
      if (d < min)
      {
        cluster_index = j;
        min = d;
      }
    }

    // Update data
    if (cluster_index != d_cluster_indices[i])
    {
      *changed = true;
    }
    d_cluster_indices[i] = cluster_index;
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
  // # Openmp prepping
  omp_set_num_threads(64);

  // # Allocate memory on device
  float *d_sample_x, *d_sample_y, *d_centroids_x, *d_centroids_y;
  int *d_cluster_indices;

  int n_bytes = 4 * N;
  int k_bytes = 4 * K;

  cudaMalloc((void **)&d_sample_x, n_bytes);
  cudaMalloc((void **)&d_sample_y, n_bytes);
  cudaMalloc((void **)&d_cluster_indices, n_bytes);
  cudaMalloc((void **)&d_centroids_x, k_bytes);
  cudaMalloc((void **)&d_centroids_y, k_bytes);
  checkCUDAError("Malloc error");

  // # Generate values and place in device memory
  init(d_sample_x, d_sample_y, d_cluster_indices, d_centroids_x, d_centroids_y);

  // # Begin main loop
  // ## Prepare convergence flag in host/device
  bool h_changed, *d_changed;
  cudaMalloc((void **)&d_changed, sizeof(bool));
  cudaMemset(d_changed, 0, sizeof(bool));
  // ## First iteration sample distribution
  int sh_bytes = SAMPLES_PER_THREAD * THREADS_PER_BLOCK * sizeof(float) * 2;
  distribute_elements_kernel<<<N_BLOCKS, THREADS_PER_BLOCK, sh_bytes>>>(d_sample_x,
                                                                        d_sample_y,
                                                                        d_centroids_x,
                                                                        d_centroids_y,
                                                                        d_cluster_indices,
                                                                        d_changed,
                                                                        N, K, SAMPLES_PER_THREAD);
  cudaDeviceSynchronize();
  // ## Get convergence flag from device
  cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
  int iterations;
  for (iterations = 0; h_changed && iterations < 21; iterations++)
  {
    // ## Calculate centroids
    calc_centroids(d_cluster_indices, d_centroids_x, d_centroids_y);
    // ## Reset 'changed' flags on device and host
    h_changed = false;
    cudaMemset(d_changed, 0, sizeof(bool));
    // ## Redistribute elements
    distribute_elements_kernel<<<N_BLOCKS, THREADS_PER_BLOCK, sh_bytes>>>(d_sample_x,
                                                                          d_sample_y,
                                                                          d_centroids_x,
                                                                          d_centroids_y,
                                                                          d_cluster_indices,
                                                                          d_changed,
                                                                          N, K, SAMPLES_PER_THREAD);
    cudaDeviceSynchronize();                                                                  
    // ## Get 'changed' flag from device
    cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
  }

  // # Print results
  printf("N = %d, K = %d\n", N, K);
  float h_centroids_x[K], h_centroids_y[K];
  cudaMemcpy(h_centroids_x, d_centroids_x, K * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_centroids_y, d_centroids_y, K * 4, cudaMemcpyDeviceToHost);
  for (int i = 0; i < K; i++)
  {
    printf("Center: (%1.3f, %1.3f) : Size: %d\n", h_centroids_x[i],
           h_centroids_y[i], h_cluster_size[i]);
  }
  printf("Iterations: %d\n", iterations);
}
