#pragma once

#include "ppm.hpp"

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include "chronoGPU.hpp"

float student1(const PPMBitmap& in, PPMBitmap& out, const int size);

// You may export your own class or functions to communicate data between the exercises ...