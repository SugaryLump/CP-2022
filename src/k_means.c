#include "k_means.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000000
#define K 4

int k, n_samples, seed;
coordinate *samples;
int *cluster_indices;
// cluster *clusters;
coordinate *centroids;

float *cluster_x;
float *cluster_y;
int *cluster_size;

float dist(coordinate a, coordinate b) {
  return (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y);
}

void init() {
  // Initialize randomization seed
  srand(10);

  // Create random samples
  samples = calloc(n_samples, sizeof(coordinate));
  cluster_indices = calloc(n_samples, sizeof(int));
  for (int i = 0; i < n_samples; i++) {
    samples[i].x = (float)rand() / RAND_MAX;
    samples[i].y = (float)rand() / RAND_MAX;
    cluster_indices[i] = -1;
  }

  // Initialize clusters
  cluster_size = calloc(k, sizeof(int));
  cluster_x = calloc(k, sizeof(float));
  cluster_y = calloc(k, sizeof(float));
  centroids = calloc(k, sizeof(coordinate));
  for (int i = 0; i < k; i++) {
    cluster_indices[i] = i;
    // clusters[i].centroid.cluster_index = i;
    cluster_size[i] = 1;
  }
}

void calc_centroids() {
  for (int i = 0; i < k; i++) {
    cluster_x[i] = 0;
    cluster_y[i] = 0;
    cluster_size[i] = 0;
  }

  for (int i = 0; i < n_samples; i++) {
    coordinate *c = &samples[i];
    cluster_x[cluster_indices[i]] += c->x;
    cluster_y[cluster_indices[i]] += c->y;
    cluster_size[cluster_indices[i]]++;
  }

  // Calculate centroids
  for (int i = 0; i < k; i++) {
    centroids[i].x = cluster_x[i] / cluster_size[i];
    centroids[i].y = cluster_y[i] / cluster_size[i];
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

  for (int i = 0; i < n_samples; i++) {
    // Find nearest cluster
    int cluster_index = 0;
    // Isto é mais lento for some reason... Vetoriza o cálculo da distância, mas
    // o cálculo do mínimo demora consideravelmente mais e acrescenta ~1s ao
    // tempo de execução float distances[k];

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

    float min = dist(samples[i], centroids[0]);
    for (int j = 0; j < k; j++) {
      float d = dist(samples[i], centroids[j]);
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
  n_samples = N;
  k = K;

  init();
  calc_centroids();
  int iterations = 0;
  while (distribute_elements()) {
    iterations++;
    calc_centroids();
  }
  // calc_centroids;

  printf("N = %d, K = %d\n", n_samples, k);
  for (int i = 0; i < k; i++) {
    printf("Center: (%f, %f) : Size: %d\n", centroids[i].x, centroids[i].y,
           cluster_size[i]);
  }
  printf("Iterations: %d\n", iterations);
}
