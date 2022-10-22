#include <math.h>

#ifndef K_MEANS_H
#define K_MEANS_H

struct Coordinate {
    float x;
    float y;
    int cluster_index;
} typedef coordinate;

float distance (coordinate *a, coordinate *b);

struct Cluster {
    coordinate **samples;
    coordinate *centroid;
    int size;
}typedef cluster;

void init();

void calc_centroids();

short distribute_elements();

#endif