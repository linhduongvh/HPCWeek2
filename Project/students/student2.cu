#include "student2.hpp"

/* Exercice 1.
* Here, you have to apply the Median Filter on the input image.
* Calculations have to be done using CUDA. 
* You have to measure the computation times, and to return the number of ms 
* your calculation took (do not include the memcpy).
*
* @param in: input image
* @param out: output (filtered) image
* @param size: width of the kernel 
*/
// float student2(const PPMBitmap &in, PPMBitmap &out, const int size) {
//     return 0.f;
// }
__global__ void ComputeV(uchar3 *input, float *output, int height, int width){
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	if (tidx >= width) return;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	if (tidy >= height) return;
	int tid = tidx + tidy * width; 
	output[tid] = max(max(input[tid].x, input[tid].y), input[tid].z) / 255.0f;
}

__global__ void MedianFilter(uchar3 *input, float *value, uchar3 *output, int height, int width, int wdsize) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	if (tidx >= width) return;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	if (tidy >= height) return;
	int tid = tidx + tidy * width;
    int m = wdsize / 2;
    uchar3 *neighbours = new uchar3[wdsize * wdsize];
    float neighboursValue[50] = {0};
    int c = 0;
    for (int i = tidx - m; i <= tidx + m; ++i){
      for (int j = tidy - m; j <= tidy + m; ++j){
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
        neighbours[c] = input[xlocal + ylocal * width];
        neighboursValue[c++] = value[xlocal + ylocal * width];
      }
    }
    for (int i = 0; i < wdsize * wdsize - 1; ++i){
      for (int j = 0; j < wdsize * wdsize - i - 1; ++j){
        if (neighboursValue[j] > neighboursValue[j + 1]) {
          uchar3 temp = neighbours[j];
          neighbours[j] = neighbours[j + 1];
          neighbours[j + 1] = temp;
          float tempValue = neighboursValue[j];
          neighboursValue[j] = neighboursValue[j + 1];
          neighboursValue[j + 1] = tempValue;
        }
      }
    }
    output[tid] = neighbours[wdsize * wdsize / 2]; 
    delete[] neighboursValue;
    delete[] neighbours;
}

float student2(const PPMBitmap &input, PPMBitmap &output, const int size) {
  int width = input.getWidth();
  int height = input.getHeight();
  int totalPixels = width * height;
  uchar3 *inputUchar3 = (uchar3*) malloc(totalPixels * sizeof(uchar3));
  uchar3 *devInput;
  float *devValue;
  uchar3 *devOutput;
  dim3 blockSize = dim3(32, 32);
  dim3 gridSize = dim3((width + 32 - 1) / 32, (height + 32 - 1) / 32);
  ChronoGPU chrGPU;
  for (int i = 0; i < width; ++i)
  {
    for (int j = 0; j < height; ++j)
    {
      PPMBitmap::RGBcol pixel = input.getPixel(i, j);
      inputUchar3[i + j * width] = make_uchar3(pixel.r, pixel.g, pixel.b);
    }
  }
  cudaMalloc(&devInput, totalPixels * sizeof(uchar3));
  cudaMalloc(&devOutput, totalPixels * sizeof(uchar3));
  cudaMalloc(&devValue, totalPixels * sizeof(float));
  cudaMemcpy(devInput, inputUchar3, totalPixels * sizeof(uchar3), cudaMemcpyHostToDevice);
  chrGPU.start();
  ComputeV<<<gridSize, blockSize>>>(devInput, devValue, height, width);
  MedianFilter<<<gridSize, blockSize>>>(devInput, devValue, devOutput, height, width, size);
  chrGPU.stop();
  uchar3 *outputUchar3 = (uchar3 *) malloc(totalPixels * sizeof(uchar3));
  cudaMemcpy(outputUchar3, devOutput, totalPixels * sizeof(uchar3), cudaMemcpyDeviceToHost);
  for (int i = 0; i < width; ++i){
    for (int j = 0; j < height; ++j){
      uchar3 pixel = outputUchar3[i + j * width];
      output.setPixel(i, j, PPMBitmap::RGBcol(pixel.x, pixel.y, pixel.z));
    }
  }
  cudaFree(devInput);
  cudaFree(devOutput);
  cudaFree(devValue);
  return chrGPU.elapsedTime();
}