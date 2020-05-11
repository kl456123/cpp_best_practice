#ifndef OPENGL_CORE_CONTEXT_H_
#define OPENGL_CORE_CONTEXT_H_
#include<vector>
#include <memory>

#include "opengl/core/opengl.h"
#include "opengl/core/tensor.h"
#include "opengl/core/buffer.h"

namespace opengl{
    class Allocator;
    class Context{
        public:
            Context(Allocator* allocator);
            Context():Context(nullptr){}
            void Compute(std::initializer_list<size_t> dim_sizes);

            void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Tensor* device_tensor);
            void CopyDeviceTensorToCPU(const Tensor* device_tensor, Tensor* cpu_tensor);

            // for now convert it in host memory instead of device memory
            // TODO(breakpoint) add 1d representation in fragment shader to support nhwc format
            // buffer in nhwc layout, texture in (nh, w, c4, 4) layout
            void ConvertTensorNHWCToNHWC4(const Tensor* nhwc_tensor, void** nhwc4_data);
            void ConvertTensorNHWC4ToNHWC(void* out, Tensor* tensor);
            void ConvertTensorNCHWToHWN4C4(const Tensor* tensor, void** out);
            void ConvertTensorNCHWToNHWC4(const Tensor* tensor, void** out);

            // primitive apis operate buffer and image
            void CopyImageToBuffer(Texture* texture, Buffer* buffer);
            void CopyBufferToImage(Texture* texture, Buffer* buffer);

            void CopyCPUBufferToDevice(Buffer* buffer, void* buffer_cpu);
            void CopyDeviceBufferToCPU(Buffer* buffer, void* buffer_cpu);
            void Finish(){glFlush();}
        private:
            // used to allocator new buffer or texture duration runtime
            Allocator* allocator_;

            // used to copy
            std::unique_ptr<ShaderBuffer> temp_buffer_;// owned

            // common compute shader
            const char* kImage2buffer_name_ = "../opengl/examples/gpgpu/image2buffer.glsl";
            const char* kBuffer2image_name_ = "../opengl/examples/gpgpu/buffer2image.glsl";
    };

    Context* GetContext();

}//namespace opengl


#endif
