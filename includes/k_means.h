#include <math.h>

#ifndef K_MEANS_H
#define K_MEANS_H

typedef struct Coordinate {
  float x;
  float y;
  int cluster_index;
} coordinate;

float distance(coordinate *a, coordinate *b);

typedef struct Cluster {
  coordinate centroid;
  float sum_x;
  float sum_y;
  int size;
} cluster;

void init();

void calc_centroids();

short distribute_elements();

#endif
