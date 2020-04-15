#include <iostream>

#include <cuda.h>



int main(){
    size_t num_bytes = 1<<10;
    int arraysize = 1000;

    float *device_data=NULL;
    cudaMalloc(&device_data, num_bytes);

    int *a_dev;
    int *b_dev;
    int *c_dev;

    cudaMalloc((void**) &a_dev, arraysize*sizeof(int));
    cudaMalloc((void**) &b_dev, arraysize*sizeof(int));
    cudaMalloc((void**) &c_dev, arraysize*sizeof(int));
    return 0;
}
