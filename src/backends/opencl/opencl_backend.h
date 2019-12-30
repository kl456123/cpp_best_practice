#ifndef OPENCL_BACKEND_H_
#define OPENCL_BACKEND_H_
#include <iostream>
#include <memory>
#include <map>
#include "context.h"
#include "core/backend.h"
#include "core/pool.h"


class Tensor;

class OpenCLPool final: public Pool{
    public:
        void* Malloc(size_t size, int alignment)override;
};

class OpenclBackend : public Backend{
    public:
        OpenclBackend(Backend::ForwardType type);
        virtual ~OpenclBackend(){}

        bool mAllocateBuffer(const size_t kSize, cl::Memory*& out_buffer);

        bool mAllocateImage(const int kHeight, const int kWidth, const float* kImageDataPtr, cl::Memory*& image);

        bool mReadBufferToHost(const cl::Buffer* buffer, int size, void* dst_host);

        bool mWriteBufferToDevice(const void* src_host, int size, cl::Buffer* buffer);

        bool mMapBufferToHost(const cl::Buffer* buffer, int size, void* dst_host);

        bool mMapHostToBuffer(const void* src_host, int size, cl::Buffer* buffer);

        void Clear()override;
        void Alloc(Tensor* )override;
        void Recycle(Tensor* )override;

        Context* runtime_ptr(){
            return mOpenCLRuntime.get();
        }
    private:
        std::shared_ptr<Context> mOpenCLRuntime;
        std::map<cl::Memory*, std::shared_ptr<cl::Memory>> mMemoryObjectsMap;
        cl_mem_flags mFlags;
};




#endif
