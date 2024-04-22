#include<iostream>
#include<omp.h>

using namespace std;

int minval(int arr[], int n){
  int minval = arr[0];
  #pragma omp parallel for reduction(min : minval)
    for(int i = 0; i < n; i++){
      if(arr[i] < minval) minval = arr[i];
    }
  return minval;
}

int maxval(int arr[], int n){
  int maxval = arr[0];
  #pragma omp parallel for reduction(max : maxval)
    for(int i = 0; i < n; i++){
      if(arr[i] > maxval) maxval = arr[i];
    }
  return maxval;
}

int sum(int arr[], int n){
  int sum = 0;
  #pragma omp parallel for reduction(+ : sum)
    for(int i = 0; i < n; i++){
      sum += arr[i];
    }
  return sum;
}

int average(int arr[], int n){
  return (double)sum(arr, n) / n;
}

int main(){
  int n;
  cout << "Enter the size of the array: ";
  cin >> n;
  int arr[n];
  cout << "Enter the elements of the array: ";
  for (int i = 0; i < n; ++i) {
    cin >> arr[i];
  }
  cout << "The minimum value is: " << minval(arr, n) << '\n';
  cout << "The maximum value is: " << maxval(arr, n) << '\n';
  cout << "The summation is: " << sum(arr, n) << '\n';
  cout << "The average is: " << average(arr, n) << '\n';
  return 0;
}








!nvcc --version
!pip install git+https://github.com/afnan47/cuda.git
%load_ext nvcc_plugi


%%writefile operation.cu

#include <iostream>
#include <cuda_runtime.h>
#include <algorithm> // For min and max operations
#include <vector>

using namespace std;

// CUDA device function for reduction to find minimum value
__global__ void min_reduction(int* d_arr, int* d_result, int size) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (global_id < size) {
        sdata[tid] = d_arr[global_id];
    } else {
        sdata[tid] = INT_MAX;
    }
    __syncthreads();

    // Reduction to find minimum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && global_id + stride < size) {
            sdata[tid] = min(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    // Store the result from the first thread in each block
    if (tid == 0) {
        d_result[blockIdx.x] = sdata[0];
    }
}

// CUDA device function for reduction to find maximum value
__global__ void max_reduction(int* d_arr, int* d_result, int size) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (global_id < size) {
        sdata[tid] = d_arr[global_id];
    } else {
        sdata[tid] = INT_MIN;
    }
    __syncthreads();

    // Reduction to find maximum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && global_id + stride < size) {
            sdata[tid] = max(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    // Store the result from the first thread in each block
    if (tid == 0) {
        d_result[blockIdx.x] = sdata[0];
    }
}

// CUDA device function for reduction to find sum
__global__ void sum_reduction(int* d_arr, int* d_result, int size) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (global_id < size) {
        sdata[tid] = d_arr[global_id];
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();

    // Reduction to find sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && global_id + stride < size) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Store the result from the first thread in each block
    if (tid == 0) {
        d_result[blockIdx.x] = sdata[0];
    }
}

// Function to calculate the final reduction on the host
int final_reduction(int* d_result, int size, int (*op)(int, int)) {
    std::vector<int> h_result(size);
    cudaMemcpy(h_result.data(), d_result, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Perform final reduction on the host
    int final_result = h_result[0];
    for (int i = 1; i < size; i++) {
        final_result = op(final_result, h_result[i]);
    }

    return final_result;
}

int main() {
    int n = 5;
    int arr[] = {1, 2, 3, 4, 5};

    // Allocate memory on the GPU
    int* d_arr;
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

   // Minimum value
    int* d_min_result;
    cudaMalloc(&d_min_result, gridSize * sizeof(int));
    min_reduction<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_arr, d_min_result, n);
    int minval = final_reduction(d_min_result, gridSize, [](int a, int b) { return std::min(a, b); });


    // Maximum value
    int* d_max_result;
    cudaMalloc(&d_max_result, gridSize * sizeof(int));
    max_reduction<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_arr, d_max_result, n);
    int maxval = final_reduction(d_max_result, gridSize, [](int a, int b) { return std::max(a, b); });

    // Summation
    int* d_sum_result;
    cudaMalloc(&d_sum_result, gridSize * sizeof(int));
    sum_reduction<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_arr, d_sum_result, n);
    int sumval = final_reduction(d_sum_result, gridSize, [](int a, int b) { return a + b; });


    // Calculate average on the host
    double average = (double)sumval / n;

    std::cout << "The minimum value is: " << minval << std::endl;
    std::cout << "The maximum value is: " << maxval << std::endl;
    std::cout << "The summation is: " << sumval << std::endl;
    std::cout << "The average is: " << average << std::endl;

    cudaFree(d_arr);
    cudaFree(d_min_result);
    cudaFree(d_max_result);
    cudaFree(d_sum_result);

    return 0;
}

