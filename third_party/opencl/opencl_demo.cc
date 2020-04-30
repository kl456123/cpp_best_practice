#include <iostream>
#include <vector>
#include <fstream>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#define USE_DMA

bool BuildProgram(cl::Context context, const char* fname, cl::Program* out){
    std::ifstream sourceFile(fname);
    if(sourceFile.fail()){
        std::cout<<"Load Kernel File Failed!"<<std::endl;
        return -1;
    }
    std::string sourceCode(
            std::istreambuf_iterator<char>(sourceFile),
            (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(),
                sourceCode.length()+1));
    cl::Program program(context, source);
    program.build(context.getInfo<CL_CONTEXT_DEVICES>());


    *out = program;
    return 0;
}

bool BuildKernel(cl::Program program, const char* kernel_name, cl::Kernel* out){

    cl::Kernel kernel(program, kernel_name);
    *out = kernel;
    return true;
}

cl::CommandQueue CreateStream(cl::Context context){
    cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
    return cl::CommandQueue(context, device);
}


bool MemcpyD2H(cl::Context context, cl::Buffer gpu_src, void* host_dst, uint64_t bytes){
#ifdef USE_DMA
    auto stream = CreateStream(context);
    auto buffer_ptr = stream.enqueueMapBuffer(gpu_src, CL_TRUE, CL_MAP_READ, 0, bytes);
    memcpy(host_dst, buffer_ptr, bytes);
    stream.enqueueUnmapMemObject(gpu_src, buffer_ptr);
#else
    auto stream = CreateStream(context);
    stream.enqueueReadBuffer(gpu_src, CL_TRUE, 0, bytes, host_dst);
#endif
    return true;
}

bool MemcpyH2D(cl::Context context, void* host_src, cl::Buffer gpu_dst, uint64_t bytes){
#ifdef USE_DMA
    auto stream = CreateStream(context);
    auto buffer_ptr = stream.enqueueMapBuffer(gpu_dst, CL_TRUE, CL_MAP_WRITE, 0, bytes);
    memcpy(buffer_ptr, host_src, bytes);
    stream.enqueueUnmapMemObject(gpu_dst, buffer_ptr);
#else
    auto stream = CreateStream(context);
    stream.enqueueWriteBuffer(gpu_dst, CL_TRUE, 0, bytes, host_src);
#endif
    return true;
}


using namespace std;
int main(){
    /////////////////////////////////
    //platfrom
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    cl::Platform platform = all_platforms[0];
    std::cout<<"num of platforms: "<<all_platforms.size()<<std::endl;

    /////////////////////////////////
    // device
    std::vector<cl::Device> all_devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    cl::Device device;
    device = all_devices[0];


    //////////////////////////////
    // context and command queue
    cl::Context context({device});
    cl::CommandQueue command_queue(context, device);
    // cl::CommandQueue command_queue = cl::CommandQueue::getDefault();

    /////////////////////////////////
    // program and kernel
    const char fname[] = "../vector_add_kernel.cl";
    cl::Program program;
    cl::Kernel kernel;
    {
        cl::Program tmp_program;
        cl::Kernel tmp_kernel;
        bool res = BuildProgram(context, fname, &tmp_program);

        const char kernel_name[] = "vector_add";
        res = BuildKernel(tmp_program, kernel_name, &tmp_kernel);
        program = tmp_program;
        kernel = tmp_kernel;
        // std::cout<<"Program "<<program()<<std::endl;
    }
    // std::cout<<"Program "<<program()<<std::endl;



    /////////////////////////////////
    // Prepare CPU data And GPU data, Then Set Input And Output
    const int N = 100;
    const int bytes = N * sizeof(float);
    float *A = new float[N];
    float *B = new float[N];
    float *C = new float[N];
    const float D = 4.0;
    for(int i=0;i<N;i++){
        A[i] = 1.0;
        B[i] = 2.0;
    }

    cl_mem_flags flags = CL_MEM_READ_WRITE;
    cl::Buffer input0(context, flags, bytes);
    cl::Buffer input1(context, flags, bytes);
    cl::Buffer output(context, flags, bytes);

    // copy cpu to gpu
    MemcpyH2D(context, A, input0, bytes);
    MemcpyH2D(context, B, input1, bytes);


    kernel.setArg(0, input0);
    kernel.setArg(1, input1);
    kernel.setArg(2, output);

    // //////////////////////////////
    // // launch kernel
    cl::NDRange gws = {N};
    cl::NDRange lws = {1};
    command_queue.enqueueNDRangeKernel(kernel, cl::NullRange,
            gws, lws);

    // TODO do we need it, or where should we place it
    command_queue.finish();
    // copy gpu to cpu
    MemcpyD2H(context, output, C, bytes);


    ///////////////////////////
    // test result
    // print first
    for(int i=0;i<10;++i){
        std::cout<<C[i]<<std::endl;
    }

    return 0;
}
