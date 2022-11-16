#include "k_means.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

int N;
int K;
int n_threads = 1;

int *cluster_indices;
float *centroids_x;
float *centroids_y;

float *sample_x;
float *sample_y;

int *cluster_size;

float dist(int a, int b) {
  float a_x = sample_x[a];
  float a_y = sample_y[a];

  float b_x = centroids_x[b];
  float b_y = centroids_y[b];
  return (b_x - a_x) * (b_x - a_x) + (b_y - a_y) * (b_y - a_y);
}

void init() {
  // Initialize randomization seed
  srand(10);

  // Create random samples
  sample_x = malloc(N * sizeof(float));
  sample_y = malloc(N * sizeof(float));
  cluster_indices = malloc(N * sizeof(int));
  for (int i = 0; i < N; i++) {
    sample_x[i] = (float)rand() / RAND_MAX;
    sample_y[i] = (float)rand() / RAND_MAX;
    cluster_indices[i] = -1;
  }

  // Initialize clusters
  cluster_size = malloc(K * sizeof(int));
  centroids_x = malloc(K * sizeof(float));
  centroids_y = malloc(K * sizeof(float));
  for (int i = 0; i < K; i++) {
    centroids_x[i] = sample_x[i];
    centroids_y[i] = sample_y[i];
  }
}

void calc_centroids() {
  float cluster_x[K], cluster_y[K];
  int p_cluster_size[K];
  //float t_cluster_x[K], t_cluster_y[K];
  //int t_cluster_size[K];

#pragma omp smd
  for (int i = 0; i < K; i++) {
    cluster_x[i] = 0;
    cluster_y[i] = 0;
    p_cluster_size[i] = 0;
  }
  // for (int i = 0; i < K; i++) {
  //   t_cluster_x[i] = 0;
  //   t_cluster_y[i] = 0;
  //   t_cluster_size[i] = 0;
  // }

#pragma omp parallel for reduction(+:cluster_x, cluster_y, p_cluster_size)
  for (int i = 0; i < N; i++) {
    cluster_x[cluster_indices[i]] += sample_x[i];
    cluster_y[cluster_indices[i]] += sample_y[i];
    p_cluster_size[cluster_indices[i]]++;
  }
  // for (int i = 0; i < N; i++) {
  //   t_cluster_x[cluster_indices[i]] += sample_x[i];
  //   t_cluster_y[cluster_indices[i]] += sample_y[i];
  //   t_cluster_size[cluster_indices[i]]++;
  // }

  // for (int i = 0; i < K; i++) {
  //   if (t_cluster_x[i] != cluster_x[i]) {
  //         printf("ERROR in cluster %d:\nParallel:\t(%f, %f), %d\nSequential:\t(%f, %f), %d\n\n", i, cluster_x[i], cluster_y[i], p_cluster_size[i], t_cluster_x[i], t_cluster_y[i], t_cluster_size[i]);
  //       }
  // }

  // Calculate centroids
  for (int i = 0; i < K; i++) {
    cluster_size[i] = p_cluster_size[i];
    centroids_x[i] = cluster_x[i] / cluster_size[i];
    centroids_y[i] = cluster_y[i] / cluster_size[i];
  }
}

bool distribute_elements() {
  bool changed = false;
#pragma omp parallel for reduction(||:changed)
  for (int i = 0; i < N; i++) {
    // Find nearest cluster
    int cluster_index = 0;

    float min = dist(i, 0);
    for (int j = 1; j < K; j++) {
      float d = dist(i, j);
      if (d < min) {
        cluster_index = j;
        min = d;
      }
    }

    // Update data
    if (cluster_index != cluster_indices[i]) {
      changed = true;
    }
    cluster_indices[i] = cluster_index;
  }

  return changed;
}

int main(int argc, char **argv) {
  if (argc > 2) {
    sscanf(argv[1], "%d", &N);
    sscanf(argv[2], "%d", &K);
    if (argc > 3) {
      sscanf(argv[3], "%d", &n_threads);
      omp_set_num_threads(n_threads);
    }
  }
  else {
    printf("Usage: ./k_means <#samples> <#clusters> <#threads>");
    return 0;
  }

  init();

  int iterations = 0;  
  for (iterations = 0; iterations < 21 && distribute_elements(); iterations++) {
    calc_centroids();
  }

  printf("N = %d, K = %d\n", N, K);
  for (int i = 0; i < K; i++) {
    printf("Center: (%1.3f, %1.3f) : Size: %d\n", centroids_x[i], centroids_y[i],
           cluster_size[i]);
  }
  printf("Iterations: %d\n", iterations);
}
