#include "k_means.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000000
#define K 4

int k, n_samples, seed;
coordinate *samples;
cluster *clusters;

float dist(coordinate a, coordinate b) {
  return (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y);
}

void init() {
  // Initialize randomization seed
  srand(10);

  // Create random samples
  samples = calloc(n_samples, sizeof(coordinate));
  for (int i = 0; i < n_samples; i++) {
    samples[i].x = (float)rand() / RAND_MAX;
    samples[i].y = (float)rand() / RAND_MAX;
    samples[i].cluster_index = -1;
  }

  // Initialize clusters
  clusters = calloc(k, sizeof(cluster));
  for (int i = 0; i < k; i++) {
    samples[i].cluster_index = i;
    clusters[i].centroid.cluster_index = i;
    clusters[i].size = 1;
  }
}

void calc_centroids() {
  for (int i = 0; i < k; i++) {
    clusters[i].sum_x = 0;
    clusters[i].sum_y = 0;
    clusters[i].size = 0;
  }

  for (int i = 0; i < n_samples; i++) {
    coordinate *c = &samples[i];
    clusters[c->cluster_index].sum_x += c->x;
    clusters[c->cluster_index].sum_y += c->y;
    clusters[c->cluster_index].size++;
  }

  // Calculate centroids
  for (int i = 0; i < k; i++) {
    clusters[i].centroid.x = clusters[i].sum_x / clusters[i].size;
    clusters[i].centroid.y = clusters[i].sum_y / clusters[i].size;
  }

  //   for (int j = 0; j < clusters[i].size; j++) {
  //     sum_x += clusters[i].samples[j]->x;
  //     sum_y += clusters[i].samples[j]->y;
  //   }
  //   clusters[i].centroid->x = sum_x / clusters[i].size;
  //   clusters[i].centroid->y = sum_y / clusters[i].size;
  // }
}

short distribute_elements() {
  short changed = 0;

  // Reset cluster sizes;
  for (int i = 0; i < k; i++) {
    clusters[i].size = 0;
  }

  float distances[K];
  for (int i = 0; i < n_samples; i++) {
    // Find nearest cluster
    int cluster_index = 0;
    float min = dist(samples[i], clusters[0].centroid);
    for (int j = 0; j < k; j++) {
      distances[j] = dist(samples[i], clusters[j].centroid);
      float d = dist(samples[i], clusters[j].centroid);
      if (d < min) {
        cluster_index = j;
        min = d;
      }
    }

    // Update data
    if (cluster_index != samples[i].cluster_index) {
      changed = 1;
    }
    samples[i].cluster_index = cluster_index;
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
    cluster c = clusters[i];
    printf("Center: (%f, %f) : Size: %d\n", c.centroid.x, c.centroid.y, c.size);
  }
  printf("Iterations: %d\n", iterations);

  return 1;
}
