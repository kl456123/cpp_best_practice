#include "opencl_backend.h"



OpenclBackend::OpenclBackend(){
    mOpenCLRuntime.reset(new Context);
    mFlags = CL_MEM_READ_WRITE;
}


template<typename DTYPE>
bool OpenclBackend::mAllocateBuffer(const int kSize, cl::Memory*& out_buffer){
    out_buffer = (cl::Memory*)(new cl::Buffer(mOpenCLRuntime->context(), mFlags, sizeof(DTYPE)*kSize));

    std::shared_ptr<cl::Memory> buffer;
    buffer.reset(out_buffer);

    mMemoryObjectsMap.insert(std::make_pair(buffer.get(), buffer));
    out_buffer = buffer.get();
    return true;
}

template<typename DTYPE>
bool OpenclBackend::mAllocateImage(const int kHeight, const int kWidth, const float* kImageDataPtr, cl::Memory*& image){
    std::shared_ptr<cl::Memory> image_ptr;
    image_ptr.reset(new cl::Image2D(mOpenCLRuntime->context(), mFlags, \
                cl::ImageFormat(CL_R, CL_FLOAT), kWidth, kHeight, 0, (void*)((float*)kImageDataPtr)));
    mMemoryObjectsMap.emplace(image_ptr.get(),image_ptr);
    image = image_ptr.get();
    return true;
}

template<typename DTYPE>
bool OpenclBackend::mReadBufferToHost(const cl::Buffer* buffer, int size, DTYPE* dst_host){
    mOpenCLRuntime->command_queue().enqueueReadBuffer(*buffer, CL_TRUE, 0, sizeof(DTYPE)*size, dst_host);
    return true;
}

template<typename DTYPE>
bool OpenclBackend::mMapBufferToHost(const cl::Buffer* buffer, int size, DTYPE* dst_host){
    auto buffer_ptr = mOpenCLRuntime->command_queue().enqueueMapBuffer(*buffer, CL_TRUE, CL_MAP_READ, 0, size);
    memcpy( dst_host, buffer_ptr,size);
    mOpenCLRuntime->command_queue().enqueueUnmapMemObject(*buffer, buffer_ptr);
    return true;
}

template<typename DTYPE>
bool OpenclBackend::mMapHostToBuffer(const DTYPE* src_host, int size, cl::Buffer* buffer){
    auto buffer_ptr = mOpenCLRuntime->command_queue().enqueueMapBuffer(*buffer, CL_TRUE, CL_MAP_WRITE, 0, size);
    memcpy(buffer_ptr, src_host, size);
    mOpenCLRuntime->command_queue().enqueueUnmapMemObject(*buffer, buffer_ptr);
    return true;
}

template<typename DTYPE>
bool OpenclBackend::mWriteBufferToDevice(const DTYPE* src_host,int size, cl::Buffer* buffer){
    mOpenCLRuntime->command_queue().enqueueWriteBuffer(*buffer, CL_TRUE, 0, sizeof(DTYPE)*size, src_host);
    return true;
}



template bool OpenclBackend::mAllocateBuffer<float>(int kSize, cl::Memory*& out_buffer);
template bool OpenclBackend::mReadBufferToHost<float>(const cl::Buffer*, int, float*);
template bool OpenclBackend::mWriteBufferToDevice<float>(const float*, int, cl::Buffer*);
template bool OpenclBackend::mMapHostToBuffer<float>(const float*, int, cl::Buffer*);
template bool OpenclBackend::mMapBufferToHost<float>(const cl::Buffer*, int, float*);




