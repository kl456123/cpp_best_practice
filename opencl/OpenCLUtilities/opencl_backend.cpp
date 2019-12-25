#include "opencl_backend.h"



OpenclBackend::OpenclBackend(){
    mOpenCLRuntime.reset(new Context);
}


template<typename DTYPE>
bool OpenclBackend::mAllocateBuffer(const int kSize, cl::Memory*& out_buffer){
    auto buffer = std::make_shared<cl::Buffer>(mOpenCLRuntime->context(), CL_MEM_WRITE_ONLY, sizeof(DTYPE)*kSize);
    mMemoryObjectsMap.emplace(buffer.get(), buffer);
    out_buffer = buffer.get();
    return true;
}

template<typename DTYPE>
bool OpenclBackend::mAllocateImage(const int kHeight, const int kWidth, const float* kImageDataPtr, cl::Memory*& image){
    auto clImage = std::make_shared<cl::Image2D>(mOpenCLRuntime->context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, \
            cl::ImageFormat(CL_R, CL_FLOAT), kWidth, kHeight, 0, (void*)((float*)kImageDataPtr));
    image = clImage.get();
    return true;
}




