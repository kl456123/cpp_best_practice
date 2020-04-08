#ifndef CONTEXT_H_
#define CONTEXT_H_
#include<vector>
#include <memory>

#include "opengl.h"
#include "tensor.h"
#include "buffer.h"

class Allocator;
class Context{
    public:
        Context(Allocator* allocator);
        Context():Context(nullptr){}
        void Compute(std::initializer_list<size_t> dim_sizes);

        void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Tensor* device_tensor);
        void CopyDeviceTensorToCPU(const Tensor* device_tensor, Tensor* cpu_tensor);
        void Finish(){glFlush();}
    private:
        // used to allocator new buffer or texture duration runtime
        Allocator* allocator_;

        // used to copy
        std::unique_ptr<ShaderBuffer> temp_buffer_;// owned
};




#endif
