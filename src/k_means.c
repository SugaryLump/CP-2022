#include "k_means.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000000
#define K 4

// coordinate *samples;
int *cluster_indices;
// cluster *clusters;
// coordinate *centroids;
float *centroids_x;
float *centroids_y;

float *sample_x;
float *sample_y;

float *cluster_x;
float *cluster_y;
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
  cluster_x = malloc(K * sizeof(float));
  cluster_y = malloc(K * sizeof(float));
  // centroids = malloc(K *sizeof(coordinate));
  centroids_x = malloc(K * sizeof(float));
  centroids_y = malloc(K * sizeof(float));
  for (int i = 0; i < K; i++) {
    cluster_indices[i] = i;
    // clusters[i].centroid.cluster_index = i;
    // cluster_size[i] = 1;
  }
}

void calc_centroids() {
#pragma omp smd
  for (int i = 0; i < K; i++) {
    cluster_x[i] = 0;
    cluster_y[i] = 0;
    cluster_size[i] = 0;
  }

  for (int i = 0; i < N; i++) {
    cluster_x[cluster_indices[i]] += sample_x[i];
    cluster_y[cluster_indices[i]] += sample_y[i];
    cluster_size[cluster_indices[i]]++;
  }

  // Calculate centroids
  for (int i = 0; i < K; i++) {
    centroids_x[i] = cluster_x[i] / cluster_size[i];
    centroids_y[i] = cluster_y[i] / cluster_size[i];
  }

  //   for (int j = 0; j < clusters[i].size; j++) {
  //     sum_x += clusters[i].samples[j]->x;
  //     sum_y += clusters[i].samples[j]->y;
  //   }
  //   clusters[i].centroid->x = sum_x / clusters[i].size;
  //   clusters[i].centroid->y = sum_y / clusters[i].size;
  // }
}

bool distribute_elements() {
  bool changed = false;

  for (int i = 0; i < N; i++) {
    // Find nearest cluster
    int cluster_index = 0;
    // Isto é mais lento for some reason... Vetoriza o cálculo da distância, mas
    // o cálculo do mínimo demora consideravelmente mais e acrescenta ~1s ao
    // tempo de execução

    // float distances[k];

    // for (int j = 0; j < k; j++) {
    //   distances[j] = dist(samples[i], centroids[j]);
    // }

    // float min = distances[0];
    // for (int j = 1; j < k; j++) {
    //   if (distances[j] < min) {
    //     cluster_index = j;
    //     min = distances[j];
    //   }
    // }

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
    // clusters[cluster_index].samples[clusters[cluster_index].size] =
    // &samples[i]; clusters[cluster_index].size++;
  }

  return changed;
}

int main(int argc, char **argv) {
  init();
  calc_centroids();
  int iterations = 0;
  while (distribute_elements()) {
    iterations++;
    calc_centroids();
  }

  printf("N = %d, K = %d\n", N, K);
  for (int i = 0; i < K; i++) {
    printf("Center: (%f, %f) : Size: %d\n", centroids_x[i], centroids_y[i],
           cluster_size[i]);
  }
  printf("Iterations: %d\n", iterations);
}
