#include "opencl_backend.h"



OpenclBackend::OpenclBackend(){
    mOpenCLRuntime.reset(new Context);
}


template<typename DTYPE>
bool OpenclBackend::mAllocateBuffer(const int kSize, cl::Memory*& out_buffer){
    out_buffer = (cl::Memory*)(new cl::Buffer(mOpenCLRuntime->context(), CL_MEM_WRITE_ONLY, sizeof(DTYPE)*kSize));

    std::shared_ptr<cl::Memory> buffer;
    buffer.reset(out_buffer);

    mMemoryObjectsMap.insert(std::make_pair(buffer.get(), buffer));
    out_buffer = buffer.get();
    return true;
}

template<typename DTYPE>
bool OpenclBackend::mAllocateImage(const int kHeight, const int kWidth, const float* kImageDataPtr, cl::Memory*& image){
    std::shared_ptr<cl::Memory> image_ptr;
    image_ptr.reset(new cl::Image2D(mOpenCLRuntime->context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, \
            cl::ImageFormat(CL_R, CL_FLOAT), kWidth, kHeight, 0, (void*)((float*)kImageDataPtr)));
    mMemoryObjectsMap.emplace(image_ptr.get(),image_ptr);
    image = image_ptr.get();
    return true;
}

template<typename DTYPE>
bool OpenclBackend::mCopyBufferToHost(const cl::Buffer* buffer, int size, DTYPE* dst_host){
    mOpenCLRuntime->command_queue().enqueueReadBuffer(*buffer, CL_TRUE, 0, sizeof(DTYPE)*size, dst_host);
    return true;
}



template bool OpenclBackend::mAllocateBuffer<float>(int kSize, cl::Memory*& out_buffer);
template bool OpenclBackend::mCopyBufferToHost<float>(const cl::Buffer*, int, float*);




