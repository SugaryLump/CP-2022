#include <math.h>
#include <stdbool.h>
#include <iostream>

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

void checkCUDAError (const char *msg) {
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		std::cerr << "Cuda error: " << msg << ", " << cudaGetErrorString( err) << std::endl;
		exit(-1);
	}
}

#endif
