#include <stdio.h>
#include <string.h>

#include <CL/cl.h>
#define MAX_SOURCE_SIZE (0x100000)

cl_int BuildProgram(cl_context context, const char* fname, cl_program* out){
    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen(fname, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    cl_int ret;
    cl_program program = clCreateProgramWithSource(context, 1,
            (const char **)&source_str, (const size_t *)&source_size, &ret);

    // Build the program
    size_t size;
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);
    cl_device_id* devices = (cl_device_id*)alloca(size);

    ret = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
    *out = program;

    return ret;
}



cl_command_queue CreateStream(cl_context context){
    size_t size;
    cl_int ret;
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);
    cl_device_id* devices = (cl_device_id*)alloca(size);
    cl_command_queue stream = clCreateCommandQueue(context, devices[0], 0, &ret);
    return stream;
}

cl_int MemcpyD2H(cl_context context, cl_mem gpu_src_ptr, void* host_dst, uint64_t bytes){
    // c api
    cl_command_queue stream = CreateStream(context);
    cl_int error;
    cl_mem buffer_ptr = clEnqueueMapBuffer(stream, gpu_src_ptr, CL_TRUE, CL_MAP_READ,
            0, bytes, 0, 0, 0, &error);
    memcpy(host_dst, buffer_ptr, bytes);

    clEnqueueUnmapMemObject(stream, gpu_src_ptr, buffer_ptr, 0, 0, 0);
    return error;
}



cl_int MemcpyH2D(cl_context context, void* host_src, cl_mem gpu_dst_ptr, uint64_t bytes){
    // c api
    cl_int ret;
    cl_command_queue stream = CreateStream(context);
    cl_mem buffer_ptr = clEnqueueMapBuffer(stream, gpu_dst_ptr, CL_TRUE, CL_MAP_READ,
            0, bytes, 0, 0, 0, &ret);
    memcpy(buffer_ptr, host_src, bytes);
    clEnqueueUnmapMemObject(stream, gpu_dst_ptr, buffer_ptr, 0, 0, 0);
    return ret;
}



int main(){
    /////////////////////////////////
    //platfrom
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);


    printf("num of platforms: %d\n", ret_num_platforms);

    /////////////////////////////////
    // device
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1,
            &device_id, &ret_num_devices);


    //////////////////////////////
    // context and command queue
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    /////////////////////////////////
    // program and kernel
    const char fname[] = "../vector_add_kernel.cl";
    cl_program program;
    BuildProgram(context, fname, &program);



    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);



    /////////////////////////////////
    // Prepare CPU data And GPU data, Then Set Input And Output
    const int N = 100;
    const int bytes = N * sizeof(float);
    float *A = (float*)malloc(bytes);
    float *B = (float*)malloc(bytes);
    float* C = (float*)malloc(bytes);
    for(int i=0;i<N;i++){
        A[i] = 1.0;
        B[i] = 2.0;
    }

    cl_mem_flags flags = CL_MEM_READ_WRITE;
    cl_mem input0 = clCreateBuffer(context, flags,
            bytes, NULL, &ret);
    cl_mem input1 = clCreateBuffer(context, flags,
            bytes, NULL, &ret);
    cl_mem output = clCreateBuffer(context, flags,
            bytes, NULL, &ret);

    // copy cpu to gpu
    // write/read
    ret = clEnqueueWriteBuffer(command_queue, input0, CL_TRUE, 0,
            bytes, A, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, input1, CL_TRUE, 0,
            bytes, B, 0, NULL, NULL);

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input0);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&input1);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&output);
    // map/unmap
    /* MemcpyH2D(context, B, input0, bytes); */
    // MemcpyH2D(context, B, input1, bytes);


    // //////////////////////////////
    // // launch kernel
    size_t global_item_size = N; // Process the entire lists
    size_t local_item_size = 1; // Process in groups of 64
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
            &global_item_size, &local_item_size, 0, NULL, NULL);

    // copy gpu to cpu
    // write/read
    ret = clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0,
            bytes, C, 0, NULL, NULL);


    ///////////////////////////
    // test result
    // print first
    for(int i=0;i<10;++i){
        printf("%f\n", C[i]);
    }

    return 0;
}
