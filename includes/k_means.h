#include <math.h>
#include <stdbool.h>

#ifndef K_MEANS_H
#define K_MEANS_H

typedef struct Coordinate {
  float x;
  float y;
} coordinate;

float distance(coordinate *a, coordinate *b);

typedef struct Cluster {
  float sum_x;
  float sum_y;
  int size;
} cluster;

#endif
