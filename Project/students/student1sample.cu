#include "student1.hpp"

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

/* Exercice 1.
* Here, you have to apply the Median Filter on the input image.
* Calculations have to be done using Thrust on device.
* You have to measure the computation times, and to return the number of ms
* your calculation took (do not include the memcpy).
*
* @param in: input image
* @param out: output (filtered) image
* @param size: width of the kernel
*/

class HsvFloat3Functor : public thrust::unary_function<uchar3, float3> {
public:
  __device__ float3 operator()(const uchar3 rgbColor) {
    const float R = (float)( rgbColor.x ) / 256.f;
    const float G = (float)( rgbColor.y ) / 256.f;
    const float B = (float)( rgbColor.z ) / 256.f;

    const float min   = fminf( R, fminf( G, B ) );
    const float max   = fmaxf( R, fmaxf( G, B ) );
    const float delta = max - min;

    // H
    float H;
    if( delta < FLT_EPSILON )
      H = 0.f;
    else if ( max == R )
      H = 60.f * ( G - B ) / ( delta + FLT_EPSILON ) + 360.f;
    else if ( max == G )
      H = 60.f * ( B - R ) / ( delta + FLT_EPSILON ) + 120.f;
    else
      H = 60.f * ( R - G ) / ( delta + FLT_EPSILON ) + 240.f;
    while ( H >= 360.f )
      H -= 360.f ;

    // S
    float S = max < FLT_EPSILON ? 0.f : 1.f - min / max;

    // V
    float V = max;

    return make_float3(H, S, V);
  }
};

class RgbUchar3Functor : public thrust::unary_function<float3, uchar3> {
public:
  __device__ uchar3 operator()(const float3 hsvColor) {
    const float H = hsvColor.x;
    const float S = hsvColor.y;
    const float V = hsvColor.z;

    const float d = H / 60.f;
    const int hi  = (int)d % 6;
    const float f = d - (float)hi;

    const float l   = V * ( 1.f - S );
    const float m = V * ( 1.f - f * S );
    const float n = V * ( 1.f - ( 1.f - f ) * S );

    float R, G, B;

    if    ( hi == 0 )
      { R = V; G = n; B = l; }
    else if ( hi == 1 )
      { R = m; G = V; B = l; }
    else if ( hi == 2 )
      { R = l; G = V; B = n; }
    else if ( hi == 3 )
      { R = l; G = m; B = V; }
    else if ( hi == 4 )
      { R = n; G = l; B = V; }
    else
      { R = V; G = l; B = m; }

    return make_uchar3( R*256.f, G*256.f, B*256.f );
  }
};

class MedianFunctor : public thrust::unary_function<int, float3> {
public:
  const float3 *inHSV;
  const int size, width, height;
  MedianFunctor(float3 *inHSV, int size, int width, int height) : inHSV(inHSV), size(size), width(width), height(height) {}

  __device__ float3 operator()(const int index) {
    int x = index % width;
    int y = index / width;
    int shift = size / 2;

    float3 *neighboursArray = new float3[size * size];

    int count = 0;
    for (int i = x - shift; i <= x + shift; ++i)
    {
      for (int j = y - shift; j <= y + shift; ++j)
      {
        int localX, localY;
        if (i < 0)
        {
          localX = 0;
        } else if (i >= width) {
          localX = width - 1;
        } else {
          localX = i;
        }

        if (j < 0)
        {
          localY = 0;
        } else if (j >= height) {
          localY = height - 1;
        } else {
          localY = j;
        }

        neighboursArray[count++] = inHSV[localX + localY * width];
      }
    }

    for (int i = 0; i < size * size - 1; ++i)
    {
      for (int j = 0; j < size * size - i - 1; ++j)
      {
        if (neighboursArray[j].z > neighboursArray[j + 1].z) {
          float3 temp = neighboursArray[j];
          neighboursArray[j] = neighboursArray[j + 1];
          neighboursArray[j + 1] = temp;
        }
      }
    }

    float3 median = neighboursArray[size * size / 2];
    delete[] neighboursArray;
    return median;
  }
};

float student1(const PPMBitmap &in, PPMBitmap &out, const int size) {
  int width = in.getWidth();
  int height = in.getHeight();
  int totalPixels = width * height;

  uchar3 *inUchar3 = (uchar3*) malloc(totalPixels * sizeof(uchar3));
  for (int i = 0; i < width; ++i)
  {
    for (int j = 0; j < height; ++j)
    {
      PPMBitmap::RGBcol pixel = in.getPixel(i, j);
      inUchar3[i + j * width] = make_uchar3(pixel.r, pixel.g, pixel.b);
    }
  }

  thrust::device_vector<uchar3> inUchar3Device(inUchar3, inUchar3 + totalPixels);
  thrust::device_vector<float3> inHSV(totalPixels);
  thrust::device_vector<float3> hsvMedianFiltered(totalPixels);
  thrust::device_vector<uchar3> outUchar3Device(totalPixels);

  float totalTime = 0.f;
  ChronoGPU gChr;
  gChr.start();
  thrust::transform(inUchar3Device.begin(), inUchar3Device.end(), inHSV.begin(), HsvFloat3Functor());
  gChr.stop();
  totalTime += gChr.elapsedTime();

  float3 *raw_ptr = thrust::raw_pointer_cast(inHSV.data());
  MedianFunctor medianFunctor(raw_ptr, size, width, height);

  gChr.start();
  thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(totalPixels), hsvMedianFiltered.begin(), medianFunctor);
  thrust::transform(hsvMedianFiltered.begin(), hsvMedianFiltered.end(), outUchar3Device.begin(), RgbUchar3Functor());
  gChr.stop();
  totalTime += gChr.elapsedTime();

  thrust::host_vector<uchar3> outUchar3(outUchar3Device);

  for (int i = 0; i < width; ++i)
  {
    for (int j = 0; j < height; ++j)
    {
      uchar3 pixel = outUchar3[i + j * width];
      out.setPixel(i, j, PPMBitmap::RGBcol(pixel.x, pixel.y, pixel.z));
    }
  }

  return totalTime;
}
