#include "backends/opencl/opencl_backend.h"

void* OpenCLPool::Malloc(size_t size, int alignment){
    Backend *bn = ExtractBackend(Backend::ForwardType::OPENCL);
    cl::Memory* out_buffer=nullptr;
    dynamic_cast<OpenclBackend*>(bn)->mAllocateBuffer(size, out_buffer);
    return (void*)out_buffer;
}

OpenclBackend::OpenclBackend(Backend::ForwardType type):Backend(type){
    mOpenCLRuntime.reset(new Context);
    mFlags = CL_MEM_READ_WRITE;
}


bool OpenclBackend::mAllocateBuffer(const size_t kSize, cl::Memory*& out_buffer){
    out_buffer = (cl::Memory*)(new cl::Buffer(mOpenCLRuntime->context(), mFlags, kSize));

    std::shared_ptr<cl::Memory> buffer;
    buffer.reset(out_buffer);

    mMemoryObjectsMap.insert(std::make_pair(buffer.get(), buffer));
    out_buffer = buffer.get();
    return true;
}

bool OpenclBackend::mAllocateImage(const int kHeight, const int kWidth, const float* kImageDataPtr, cl::Memory*& image){
    std::shared_ptr<cl::Memory> image_ptr;
    image_ptr.reset(new cl::Image2D(mOpenCLRuntime->context(), mFlags, \
                cl::ImageFormat(CL_R, CL_FLOAT), kWidth, kHeight, 0, (void*)((float*)kImageDataPtr)));
    mMemoryObjectsMap.emplace(image_ptr.get(),image_ptr);
    image = image_ptr.get();
    return true;
}

bool OpenclBackend::mReadBufferToHost(const cl::Buffer* buffer, int size, void* dst_host){
    mOpenCLRuntime->command_queue().enqueueReadBuffer(*buffer, CL_TRUE, 0, size, dst_host);
    return true;
}

bool OpenclBackend::mMapBufferToHost(const cl::Buffer* buffer, int size, void* dst_host){
    auto buffer_ptr = mOpenCLRuntime->command_queue().enqueueMapBuffer(*buffer, CL_TRUE, CL_MAP_READ, 0, size);
    memcpy( dst_host, buffer_ptr,size);
    mOpenCLRuntime->command_queue().enqueueUnmapMemObject(*buffer, buffer_ptr);
    return true;
}

bool OpenclBackend::mMapHostToBuffer(const void* src_host, int size, cl::Buffer* buffer){
    auto buffer_ptr = mOpenCLRuntime->command_queue().enqueueMapBuffer(*buffer, CL_TRUE, CL_MAP_WRITE, 0, size);
    memcpy(buffer_ptr, src_host, size);
    mOpenCLRuntime->command_queue().enqueueUnmapMemObject(*buffer, buffer_ptr);
    return true;
}

bool OpenclBackend::mWriteBufferToDevice(const void* src_host,int size, cl::Buffer* buffer){
    mOpenCLRuntime->command_queue().enqueueWriteBuffer(*buffer, CL_TRUE, 0, size, src_host);
    return true;
}

void OpenclBackend::Clear(){}
void OpenclBackend::Alloc(Tensor* ){}
void OpenclBackend::Recycle(Tensor* ){
}





void RegisterOpenCLBackend(){
    std::shared_ptr<Backend> ptr;
    ptr.reset(new OpenclBackend(Backend::ForwardType::OPENCL));
    InsertBackend(Backend::ForwardType::OPENCL, ptr);
}
