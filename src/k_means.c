#include "k_means.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000000
#define K 4

int k, n_samples, seed;
coordinate **samples;
cluster **clusters;

float distance(coordinate *a, coordinate *b) {
  return sqrt(pow(b->x - a->x, 2) + pow(b->y - a->y, 2));
}

void init() {
  // Initialize randomization seed
  srand(10);

  // Create random samples
  samples = calloc(n_samples, sizeof(coordinate *));
  for (int i = 0; i < n_samples; i++) {
    coordinate *sample = calloc(1, sizeof(coordinate));
    sample->x = (float)rand() / RAND_MAX;
    sample->y = (float)rand() / RAND_MAX;
    sample->cluster_index = -1;
    samples[i] = sample;
  }

  // Initialize clusters
  clusters = calloc(k, sizeof(cluster *));
  for (int i = 0; i < k; i++) {
    cluster *new_cluster = calloc(1, sizeof(cluster));
    new_cluster->samples = calloc(n_samples, sizeof(coordinate *));
    new_cluster->samples[0] = samples[i];
    samples[i]->cluster_index = i;
    new_cluster->centroid = calloc(1, sizeof(coordinate));
    new_cluster->size = 1;
    clusters[i] = new_cluster;
  }
}

void calc_centroids() {
  for (int i = 0; i < k; i++) {
    float sum_x = 0;
    float sum_y = 0;
    for (int j = 0; j < clusters[i]->size; j++) {
      sum_x += clusters[i]->samples[j]->x;
      sum_y += clusters[i]->samples[j]->y;
    }
    clusters[i]->centroid->x = sum_x / clusters[i]->size;
    clusters[i]->centroid->y = sum_y / clusters[i]->size;
  }
}

short distribute_elements() {
  short changed = 0;

  // Reset cluster sizes;
  for (int i = 0; i < k; i++) {
    clusters[i]->size = 0;
  }

  for (int i = 0; i < n_samples; i++) {
    // Find nearest cluster
    int cluster_index = 0;
    float min = distance(samples[i], clusters[0]->centroid);
    for (int j = 1; j < k; j++) {
      float dist = distance(samples[i], clusters[j]->centroid);
      if (dist < min) {
        cluster_index = j;
        min = dist;
      }
    }

    // Update data
    if (cluster_index != samples[i]->cluster_index) {
      changed = 1;
    }
    samples[i]->cluster_index = cluster_index;
    cluster *c = clusters[cluster_index];
    c->samples[c->size] = samples[i];
    c->size++;
  }

  return changed;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    n_samples = N;
    k = K;
  } else {
    sscanf(argv[1], "%d", &n_samples);
    sscanf(argv[2], "%d", &k);
  }

  if (k > n_samples) {
    k = n_samples;
  }

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
    cluster *c = clusters[i];
    printf("Center: (%f, %f) : Size: %d\n", c->centroid->x, c->centroid->y,
           c->size);
  }
  printf("Iterations: %d\n", iterations);

  return 1;
}
