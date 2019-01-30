#include "student1.hpp"

/* Exercice 1.
* here, you have to apply the Median Filter on the input image.
* Calculations have to be done using Thrust on device. 
* You have to measure the computation times, and to return the number of ms 
* your calculation took (do not include the memcpy).
*
* @param in: input image
* @param out: output (filtered) image
* @param size: width of the kernel 
*/

class HSV : public thrust::unary_function<uchar3, float3> {
public:
	__device__ float3 operator()(const uchar3 rgb) {
		const float r = (float)( rgb.x ) / 256.f;
    const float g = (float)( rgb.y ) / 256.f;
    const float b = (float)( rgb.z ) / 256.f;
    const float max   = fmaxf( r, fmaxf( g, b ) );
    const float min   = fminf( r, fminf( g, b ) );
    const float delta = max - min;
//compute h
    float h;
    if( delta < FLT_EPSILON ){
    	h = 0.f;
    }else if (max==r){
    	h = 60.f * ( g - b ) / ( delta + FLT_EPSILON ) + 360.f;
    }else if (max==g){
    	h = 60.f * ( b - r ) / ( delta + FLT_EPSILON ) + 120.f;
    }else{
    	h = 60.f * ( r - g ) / ( delta + FLT_EPSILON ) + 240.f;
    }
    while (h>= 360.f){
    	h = h-360.f;
    }
//compute S
    float s = max < FLT_EPSILON ? 0.f : 1.f - min / max;
//compute v 
    float v = max;

    return make_float3(h, s, v);
  }
};

class RGB : public thrust::unary_function<float3, uchar3> {
public:
  __device__ uchar3 operator()(const float3 hsv) {
    const float h = hsv.x;
    const float s = hsv.y;
    const float v = hsv.z;
    const float d = h / 60.f;
    const int hi  = (int)fmodf(d,6.0);
    const float f = d - (float)hi;
    const float l   = v * ( 1.f - s );
    const float m = v * ( 1.f - (f * s) );
    const float n = v * ( 1.f - ( 1.f - f ) * s );
    float r, g, b;
    if ( hi == 0 ){
    	r = v; 
    	g = n; 
    	b = l; 
    }else if ( hi == 1 ){ 
    	r = m; 
    	g = v; 
    	b = l; 
    }else if ( hi == 2 ){ 
    	r = l; 
    	g = v; 
    	b = n; 
    }else if ( hi == 3 ){ 
    	r = l; 
    	g = m; 
    	b = v; 
    }else if ( hi == 4 ){ 
    	r = n; 
    	g = l; 
    	b = v; 
    }else{ 
    	r = v; 
    	g = l; 
    	b = m; 
    }
    return make_uchar3( r*256.f, g*256.f, b*256.f );
  }
};

class MedianFilter : public thrust::unary_function<int, float3> {
public:
  const float3 *hsv;
  const int size, width, height;
  MedianFilter (float3 *hsv, int size, int width, int height) : hsv(hsv), size(size), width(width), height(height) {}

  __device__ float3 operator()(const int index) {
    int x = index % width;
    int y = index / width;
    int m = size / 2;
    float3 neighbours[50] = {0};
		int c = 0;
    for (int i = x - m; i <= x + m; ++i){
      for (int j = y - m; j <= y + m; ++j){
        int xlocal, ylocal;
        if (j >= height){
          ylocal = height - 1;
        }else if (j < 0) {
          ylocal = 0;
        }else {
          ylocal = j;
        }
        if (i >= width) {
          xlocal = width - 1;
        }else if (i < 0){
          xlocal = 0;
        }else {
          xlocal = i;
        }
        neighbours[c++] = hsv[xlocal + ylocal * width];
      }
    }
    for (int i = 0; i < size * size - 1; ++i){
      for (int j = 0; j < size * size - i - 1; ++j){
        if (neighbours[j].z > neighbours[j + 1].z) {
          float3 temp = neighbours[j];
          neighbours[j] = neighbours[j + 1];
          neighbours[j + 1] = temp;
        }
      }
    }
    float3 median = neighbours[size * size / 2];
    delete[] neighbours;
    return median;
  }
};

float student1(const PPMBitmap &input, PPMBitmap &output, const int size) {
  int height = input.getHeight();
  int width = input.getWidth();
  int totalPixels = width * height;
  float time = 0.f;
  ChronoGPU chrGPU;
  uchar3 *inputUchar3 = (uchar3*) malloc(totalPixels * sizeof(uchar3));
  for (int i = 0; i < width; ++i){
    for (int j = 0; j < height; ++j){
      PPMBitmap::RGBcol pixel = input.getPixel(i, j);
      inputUchar3[i + j * width] = make_uchar3(pixel.r, pixel.g, pixel.b);
    }
  }
  thrust::device_vector<uchar3> inputDevice(inputUchar3, inputUchar3 + totalPixels);
  thrust::device_vector<float3> hsv(totalPixels);
  thrust::device_vector<float3> hsvMedianFilter(totalPixels);
  thrust::device_vector<uchar3> outputDevice(totalPixels);
  chrGPU.start();
  thrust::transform(inputDevice.begin(), inputDevice.end(), hsv.begin(), HSV());
  chrGPU.stop();
  time += chrGPU.elapsedTime();
  float3 *raw_ptr = thrust::raw_pointer_cast(hsv.data());
  MedianFilter med_filter(raw_ptr, size, width, height);
  chrGPU.start();
  thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(totalPixels), hsvMedianFilter.begin(), med_filter);
  thrust::transform(hsvMedianFilter.begin(), hsvMedianFilter.end(), outputDevice.begin(), RGB());
  chrGPU.stop();
  time += chrGPU.elapsedTime();
  thrust::host_vector<uchar3> outputUchar3(outputDevice);
  for (int i = 0; i < width; ++i){
    for (int j = 0; j < height; ++j){
      uchar3 pixel = outputUchar3[i + j * width];
      output.setPixel(i, j, PPMBitmap::RGBcol(pixel.x, pixel.y, pixel.z));
    }
  }
  return time;
}