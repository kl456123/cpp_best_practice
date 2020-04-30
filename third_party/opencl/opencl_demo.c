#include <stdio.h>
#include <string.h>

#include <CL/cl.h>
#define MAX_SOURCE_SIZE (0x100000)

#define USE_DMA

#define PRINT_ERROR(ret)                    \
    do{                                     \
        if(ret!=CL_SUCCESS){                    \
            printf("%s", getErrorString(ret));  \
        }}while(0)

const char *getErrorString(cl_int error)
{
    switch(error){
        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        default: return "UNKNOWN FAILURE";
    }
}

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
    PRINT_ERROR(ret);

    *out = program;

    return ret;
}



cl_command_queue CreateStream(cl_context context){
    size_t size;
    cl_int ret;
    cl_int err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);
    if(err!=CL_SUCCESS){
        printf("error when get context info");
        return NULL;
    }

    // Get Devices From Context
    cl_device_id* devices_id = (cl_device_id*)alloca(size);
    clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices_id, NULL);
    cl_command_queue stream = clCreateCommandQueue(context, devices_id[0], 0, &ret);
    return stream;
}

cl_int MemcpyD2H(cl_context context, cl_mem gpu_src_ptr, void* host_dst, uint64_t bytes){
    cl_command_queue stream = CreateStream(context);
    cl_int error;
#ifdef USE_DMA
    cl_mem buffer_ptr = clEnqueueMapBuffer(stream, gpu_src_ptr, CL_TRUE, CL_MAP_READ,
            0, bytes, 0, 0, 0, &error);
    memcpy(host_dst, buffer_ptr, bytes);

    clEnqueueUnmapMemObject(stream, gpu_src_ptr, buffer_ptr, 0, 0, 0);
#else
    error =  clEnqueueReadBuffer(stream, gpu_src_ptr, CL_TRUE, 0,
            bytes, host_dst, 0, NULL, NULL);
#endif
    return error;
}



cl_int MemcpyH2D(cl_context context, void* host_src, cl_mem gpu_dst_ptr, uint64_t bytes){
    cl_command_queue stream = CreateStream(context);
    // c api
    cl_int ret;
#ifdef USE_DMA

    cl_mem buffer_ptr = clEnqueueMapBuffer(stream, gpu_dst_ptr, CL_TRUE, CL_MAP_WRITE,
            0, bytes, 0, 0, 0, &ret);
    memcpy(buffer_ptr, host_src, bytes);
    clEnqueueUnmapMemObject(stream, gpu_dst_ptr, buffer_ptr, 0, 0, 0);
#else
    ret = clEnqueueWriteBuffer(stream, gpu_dst_ptr, CL_TRUE, 0,
            bytes, host_src, 0, NULL, NULL);
#endif
    return ret;
}





int main(){
    /////////////////////////////////
    //platfrom
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    PRINT_ERROR(ret);


    printf("num of platforms: %d\n", ret_num_platforms);

    /////////////////////////////////
    // device
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1,
            &device_id, &ret_num_devices);
    PRINT_ERROR(ret);

    //////////////////////////////
    // context and command queue
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    PRINT_ERROR(ret);

    /////////////////////////////////
    // program and kernel
    const char fname[] = "../vector_add_kernel.cl";
    cl_program program;
    ret = BuildProgram(context, fname, &program);
    ret = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    PRINT_ERROR(ret);



    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);
    PRINT_ERROR(ret);



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
    MemcpyH2D(context, A, input0, bytes);
    MemcpyH2D(context, B, input1, bytes);

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input0);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&input1);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&output);


    // //////////////////////////////
    // // launch kernel
    size_t global_item_size = N; // Process the entire lists
    size_t local_item_size = 1; // Process in groups of 64
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
            &global_item_size, &local_item_size, 0, NULL, NULL);
    PRINT_ERROR(ret);

    // copy gpu to cpu
    // write/read
    MemcpyD2H(context, output, C, bytes);
    PRINT_ERROR(ret);


    ///////////////////////////
    // test result
    // print first
    for(int i=0;i<10;++i){
        printf("%f\n", C[i]);
    }

    return 0;
}
