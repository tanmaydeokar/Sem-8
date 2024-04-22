//Q1) Bubblesort OpenMp

#include<iostream>
#include<omp.h>

using namespace std;

void bubble(int array[], int n){
    for (int i = 0; i < n - 1; i++){
        for (int j = 0; j < n - i - 1; j++){
            if (array[j] > array[j + 1]) swap(array[j], array[j + 1]);
        }
    }
}

void pBubble(int array[], int n){
    //Sort odd indexed numbers
    for(int i = 0; i < n; ++i){    
        #pragma omp for
        for (int j = 1; j < n; j += 2){
        if (array[j] < array[j-1])
        {
          swap(array[j], array[j - 1]);
        }
    }

    // Synchronize
    #pragma omp barrier

    //Sort even indexed numbers
    #pragma omp for
    for (int j = 2; j < n; j += 2){
      if (array[j] < array[j-1])
      {
        swap(array[j], array[j - 1]);
      }
    }
  }
}

void printArray(int arr[], int n){
    for(int i = 0; i < n; i++) cout << arr[i] << " ";
    cout << "\n";
}

int main(){
    // Set up variables
    int n;
    cout << "Enter the size of the array: ";
    cin >> n;
    int arr[n];
    int brr[n];
    double start_time, end_time;

    // Input array elements
    cout << "Enter " << n << " elements of the array: ";
    for(int i = 0; i < n; i++) cin >> arr[i];
    
    // Sequential time
    start_time = omp_get_wtime();
    bubble(arr, n);
    end_time = omp_get_wtime();     
    cout << "Sequential Bubble Sort took : " << end_time - start_time << " seconds.\n";
    printArray(arr, n);
    
    // Reset the array
    for(int i = 0; i < n; i++) brr[i] = arr[i];
    
    // Parallel time
    start_time = omp_get_wtime();
    pBubble(brr, n);
    end_time = omp_get_wtime();     
    cout << "Parallel Bubble Sort took : " << end_time - start_time << " seconds.\n";
    printArray(brr, n);
} 
  


//q2) MergeSort OpenMP

#include <iostream>
#include <omp.h>

using namespace std;

void merge(int arr[], int low, int mid, int high) {
    // Create arrays of left and right partititons
    int n1 = mid - low + 1;
    int n2 = high - mid;

    int left[n1];
    int right[n2];
    
    // Copy all left elements
    for (int i = 0; i < n1; i++) left[i] = arr[low + i];
    
    // Copy all right elements
    for (int j = 0; j < n2; j++) right[j] = arr[mid + 1 + j];
    
    // Compare and place elements
    int i = 0, j = 0, k = low;

    while (i < n1 && j < n2) {
        if (left[i] <= right[j]){
            arr[k] = left[i];
            i++;
        } 
        else{
            arr[k] = right[j];
            j++;
        }
        k++;
    }

    // If any elements are left out
    while (i < n1) {
        arr[k] = left[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = right[j];
        j++;
        k++;
    }
}

void parallelMergeSort(int arr[], int low, int high) {
    if (low < high) {
        int mid = (low + high) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                parallelMergeSort(arr, low, mid);
            }

            #pragma omp section
            {
                parallelMergeSort(arr, mid + 1, high);
            }
        }
        merge(arr, low, mid, high);
    }
}

void mergeSort(int arr[], int low, int high) {
    if (low < high) {
        int mid = (low + high) / 2;
        mergeSort(arr, low, mid);
        mergeSort(arr, mid + 1, high);
        merge(arr, low, mid, high);
    }
}

int main() {
    int n;
    cout << "Enter the size of the array: ";
    cin >> n;

    int arr[n];
    double start_time, end_time;

    // Input array elements
    cout << "Enter " << n << " elements of the array: ";
    for (int i = 0; i < n; i++) cin >> arr[i];
    
    // Measure Sequential Time
    start_time = omp_get_wtime(); 
    mergeSort(arr, 0, n - 1);
    end_time = omp_get_wtime(); 
    cout << "Time taken by sequential algorithm: " << end_time - start_time << " seconds\n";

    // Reset the array
    for (int i = 0; i < n; i++) arr[i] = i + 1;
    
    // Measure Parallel time
    start_time = omp_get_wtime(); 
    parallelMergeSort(arr, 0, n - 1);
    end_time = omp_get_wtime(); 
    cout << "Time taken by parallel algorithm: " << end_time - start_time << " seconds";
    
    return 0;
}







//Q1) Bubble CUDA

!nvcc --version
!pip install git+https://github.com/afnan47/cuda.git

%load_ext nvcc_plugin

%%writefile bubblesort.cu

#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// CUDA kernel for Bubble Sort
__global__ void bubble_sort(int* d_arr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Thread index
    for (int i = 0; i < size - 1; i++) {
        int j = idx + i; // Offset to perform the bubble sort step
        if (j < size - 1 && d_arr[j] > d_arr[j + 1]) { // Swap if out of order
            int temp = d_arr[j];
            d_arr[j] = d_arr[j + 1];
            d_arr[j + 1] = temp;
        }
        __syncthreads(); // Synchronize threads within block
    }
}

// Function for Bubble Sort on CPU
void bubble_sort_cpu(int* arr, int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - 1 - i; j++) {
            if (arr[j] > arr[j + 1]) { // Swap if out of order
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

int main() {
    // Test data
    int h_arr[] = {64, 34, 25, 12, 22, 11, 90};
    int size = sizeof(h_arr) / sizeof(h_arr[0]);

    // Bubble Sort on CPU
    auto start = std::chrono::high_resolution_clock::now();
    bubble_sort_cpu(h_arr, size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Sequential Bubble Sort took " << duration.count() << " seconds\n";

    // Copying data to the device for parallel Bubble Sort
    int* d_arr;
    cudaMalloc(&d_arr, size * sizeof(int));
    cudaMemcpy(d_arr, h_arr, size * sizeof(int), cudaMemcpyHostToDevice);

    // Bubble Sort on GPU
    start = std::chrono::high_resolution_clock::now();
    int blockSize = 256; // Threads per block
    int gridSize = (size + blockSize - 1) / blockSize; // Blocks
    bubble_sort<<<gridSize, blockSize>>>(d_arr, size);
    cudaMemcpy(h_arr, d_arr, size * sizeof(int), cudaMemcpyDeviceToHost); // Copy back to host
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Parallel Bubble Sort took " << duration.count() << " seconds\n";

    // Display sorted array
    std::cout << "Sorted Array: ";
    for (int i = 0; i < size; i++) {
        std::cout << h_arr[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_arr);

    return 0;
}

!nvcc bubblesort.cu -o bubble

!./bubble



//Q2) MergeSort
!nvcc --version
!pip install git+https://github.com/afnan47/cuda.git
%load_ext nvcc_plugin

%%writefile merge.cu

#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// CUDA kernel for Bubble Sort
__global__ void bubble_sort(int* d_arr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Thread index
    for (int i = 0; i < size - 1; i++) {
        int j = idx + i; // Offset to perform the bubble sort step
        if (j < size - 1 && d_arr[j] > d_arr[j + 1]) { // Swap if out of order
            int temp = d_arr[j];
            d_arr[j] = d_arr[j + 1];
            d_arr[j + 1] = temp;
        }
        __syncthreads(); // Synchronize threads within block
    }
}

// Function for Bubble Sort on CPU
void bubble_sort_cpu(int* arr, int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - 1 - i; j++) {
            if (arr[j] > arr[j + 1]) { // Swap if out of order
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

int main() {
    // Test data
    int h_arr[] = {64, 34, 25, 12, 22, 11, 90};
    int size = sizeof(h_arr) / sizeof(h_arr[0]);

    // Bubble Sort on CPU
    auto start = std::chrono::high_resolution_clock::now();
    bubble_sort_cpu(h_arr, size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Sequential Merge Sort took " << duration.count() << " seconds\n";

    // Copying data to the device for parallel Bubble Sort
    int* d_arr;
    cudaMalloc(&d_arr, size * sizeof(int));
    cudaMemcpy(d_arr, h_arr, size * sizeof(int), cudaMemcpyHostToDevice);

    // Bubble Sort on GPU
    start = std::chrono::high_resolution_clock::now();
    int blockSize = 256; // Threads per block
    int gridSize = (size + blockSize - 1) / blockSize; // Blocks
    bubble_sort<<<gridSize, blockSize>>>(d_arr, size);
    cudaMemcpy(h_arr, d_arr, size * sizeof(int), cudaMemcpyDeviceToHost); // Copy back to host
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Parallel Merge Sort took " << duration.count() << " seconds\n";

    // Display sorted array
    std::cout << "Sorted Array: ";
    for (int i = 0; i < size; i++) {
        std::cout << h_arr[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_arr);

    return 0;
}


!nvcc merge.cu -o merge

!./merge





