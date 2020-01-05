#ifndef OPENCL_BACKEND_H_
#define OPENCL_BACKEND_H_
#include <iostream>
#include <memory>
#include <map>
#include "context.h"
#include "core/backend.h"
#include "core/pool.h"
#include "core/tensor.h"



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

        void Alloc(Tensor* )override;
        void Recycle(Tensor* )override;
        virtual void CopyFromHostToDevice(Tensor* tensor)override;
        virtual void CopyFromDeviceToHost(Tensor* tensor);

        Context* runtime_ptr(){
            return mOpenCLRuntime.get();
        }

        bool Finish(){
            int rc = mOpenCLRuntime->command_queue().finish();
            return rc==0;
        }

    private:
        std::shared_ptr<Context> mOpenCLRuntime;
        std::map<cl::Memory*, std::shared_ptr<cl::Memory>> mMemoryObjectsMap;
        cl_mem_flags mFlags;
};




#endif
