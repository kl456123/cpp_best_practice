#ifndef OPENCL_BACKEND_H_
#define OPENCL_BACKEND_H_
#include <iostream>
#include <memory>
#include <map>
#include "context.h"
#include "register/backend.h"


class Tensor;

class OpenclBackend : public Backend{
    public:
        OpenclBackend();
        virtual ~OpenclBackend(){}

        template<typename DTYPE>
            bool mAllocateBuffer(const int kSize, cl::Memory*& out_buffer);

        template<typename DTYPE>
            bool mAllocateImage(const int kHeight, const int kWidth, const float* kImageDataPtr, cl::Memory*& image);

        template<typename DTYPE>
            bool mReadBufferToHost(const cl::Buffer* buffer, int size, DTYPE* dst_host);

        template<typename DTYPE>
            bool mWriteBufferToDevice(const DTYPE* src_host, int size, cl::Buffer* buffer);

        template<typename DTYPE>
        bool mMapBufferToHost(const cl::Buffer* buffer, int size, DTYPE* dst_host);

        template<typename DTYPE>
        bool mMapHostToBuffer(const DTYPE* src_host, int size, cl::Buffer* buffer);

        void Clear()override;
        void Alloc(const Tensor* )override;
        void Recycle(const Tensor* )override;

        Context* runtime_ptr(){
            return mOpenCLRuntime.get();
        }
    private:
        std::shared_ptr<Context> mOpenCLRuntime;
        std::map<cl::Memory*, std::shared_ptr<cl::Memory>> mMemoryObjectsMap;
        cl_mem_flags mFlags;
};




#endif
