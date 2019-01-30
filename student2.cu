#include "student2.hpp"
#include "chronoGPU.hpp"

/* Exercice 2.
* Here, you have to apply the Median Filter on the input image.
* Calculations have to be done using CUDA.
* You have to measure the computation times, and to return the number of ms
* your calculation took (do not include the memcpy).
*
* @param in: input image
* @param out: output (filtered) image
* @param size: width of the kernel
*/

__global__ void RGB2Value(uchar3 *input, float *value, int width, int height) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    value[globalId] = max(max(input[globalId].x, input[globalId].y), input[globalId].z) / 255.0f;
}

__global__ void medianFilter(uchar3 *input, float *value, uchar3 *output, int width, int height, int windowSize) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    int shift = windowSize / 2;

    // float *neighboursValueArray = new float[windowSize * windowSize];
    float neighboursValueArray[50] = {0};
    uchar3 *neighboursArray = new uchar3[windowSize * windowSize];

    int count = 0;
    for (int i = globalIdX - shift; i <= globalIdX + shift; ++i)
    {
      for (int j = globalIdY - shift; j <= globalIdY + shift; ++j)
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

        neighboursArray[count] = input[localX + localY * width];
        neighboursValueArray[count++] = value[localX + localY * width];
      }
    }

    for (int i = 0; i < windowSize * windowSize - 1; ++i)
    {
      for (int j = 0; j < windowSize * windowSize - i - 1; ++j)
      {
        if (neighboursValueArray[j] > neighboursValueArray[j + 1]) {
          float tempValue = neighboursValueArray[j];
          uchar3 temp = neighboursArray[j];

          neighboursValueArray[j] = neighboursValueArray[j + 1];
          neighboursValueArray[j + 1] = tempValue;

          neighboursArray[j] = neighboursArray[j + 1];
          neighboursArray[j + 1] = temp;
        }
      }
    }

    output[globalId] = neighboursArray[windowSize * windowSize / 2];

    delete[] neighboursArray;
    delete[] neighboursValueArray;
}

float student2(const PPMBitmap &in, PPMBitmap &out, const int size) {
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

  uchar3 *devInput;
  float *devValue;
  uchar3 *devOutput;

  cudaMalloc(&devInput, totalPixels * sizeof(uchar3));
  cudaMalloc(&devValue, totalPixels * sizeof(float));
  cudaMalloc(&devOutput, totalPixels * sizeof(uchar3));

  cudaMemcpy(devInput, inUchar3, totalPixels * sizeof(uchar3), cudaMemcpyHostToDevice);

  int blockX = 32;
  int blockY = 32;
  dim3 blockSize = dim3(blockX, blockY);
  dim3 gridSize = dim3((width + blockX - 1) / blockX, (height + blockY - 1) / blockY);

  ChronoGPU gChr;
  gChr.start();
  RGB2Value<<<gridSize, blockSize>>>(devInput, devValue, width, height);
  medianFilter<<<gridSize, blockSize>>>(devInput, devValue, devOutput, width, height, size);
  gChr.stop();

  uchar3 *outUchar3 = (uchar3 *) malloc(totalPixels * sizeof(uchar3));
  cudaMemcpy(outUchar3, devOutput, totalPixels * sizeof(uchar3), cudaMemcpyDeviceToHost);

  for (int i = 0; i < width; ++i)
  {
    for (int j = 0; j < height; ++j)
    {
      uchar3 pixel = outUchar3[i + j * width];
      out.setPixel(i, j, PPMBitmap::RGBcol(pixel.x, pixel.y, pixel.z));
    }
  }

  cudaFree(devInput);
  cudaFree(devValue);
  cudaFree(devOutput);

  return gChr.elapsedTime();
}
