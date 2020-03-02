#ifndef DLCL_UTILS_GPU_KERNEL_HELPER_H_
#define DLCL_UTILS_GPU_KERNEL_HELPER_H_
#include <CL/cl.hpp>
#include "core/error.hpp"

using GpuKernel = cl::Kernel;
using GpuStream = cl::CommandQueue;

template<typename T>
void KernelSetArgFromList(GpuKernel kernel, int i,
        T&& arg){
    kernel.setArg(i, arg);
}
template<typename FirstArg, typename ...RestArgs>
void KernelSetArgFromList(GpuKernel kernel, int i, FirstArg&& first_arg, RestArgs&&... rest_args){
    kernel.setArg(i, first_arg);
    KernelSetArgFromList(kernel, i+1, rest_args...);
}

template<typename ...Args>
Status GpuSetKernel(GpuKernel kernel, Args&&... args){
    KernelSetArgFromList(kernel, 0, args...);
    return Status::OK();
}


Status GpuLaunchKernel(GpuKernel kernel, size_t gws, size_t lws,
        GpuStream stream){
    // enqueue
    cl::NDRange global_work_size = {gws};
    cl::NDRange local_work_size = {lws};
    stream.enqueueNDRangeKernel(kernel,cl::NullRange,
            global_work_size,local_work_size);

    return Status::OK();
}


#endif
