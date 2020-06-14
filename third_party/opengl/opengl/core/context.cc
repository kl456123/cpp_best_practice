#include <cstring>
#include <glog/logging.h>

#include "opengl/core/context.h"
#include "opengl/core/program.h"
#include "opengl/utils/macros.h"
#include "opengl/core/driver.h"
#include "opengl/core/tensor_format.h"
#include "opengl/core/functor.h"


namespace opengl{
    namespace internal{
        bool IsCPUDFormat(DataFormat dformat){
            if (dformat== ::dlxnet::TensorProto::ANY
                    ||dformat== ::dlxnet::TensorProto::NHWC){
                return true;
            }
            return false;
        }
        void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Tensor* device_tensor){
            CHECK(cpu_tensor->is_host());
            CHECK(!device_tensor->is_host());

            auto cpu_shape = cpu_tensor->shape();
            auto device_shape = device_tensor->shape();
            CHECK_EQ(cpu_shape.size(), device_shape.size());
            for(int i=0;i<cpu_shape.size();++i){
                CHECK_EQ(cpu_shape[i], device_shape[i]);
            }

            // check same bytes
            CHECK_EQ(cpu_tensor->AllocatedSize(), device_tensor->AllocatedSize());

            auto texture = device_tensor->device<Texture>();
            const int width = texture->width();
            const int height = texture->height();
            GLenum format = texture->format();
            GLenum type = texture->type();
            // TODO(breakpoint) why DMA is slower than non DMA
            CopyHostToTexture(cpu_tensor->host(), width, height, device_tensor->device<Texture>()->id(),
                    format, type);
        }

    void CopyDeviceTensorToCPU(const Tensor* device_tensor, Tensor* cpu_tensor){
            CHECK(cpu_tensor->is_host());
            CHECK(!device_tensor->is_host());

            auto cpu_shape = cpu_tensor->shape();
            auto device_shape = device_tensor->shape();
            CHECK_EQ(cpu_shape.size(), device_shape.size());
            for(int i=0;i<cpu_shape.size();++i){
                CHECK_EQ(cpu_shape[i], device_shape[i]);
            }

            // check same bytes
            CHECK_EQ(cpu_tensor->AllocatedSize(), device_tensor->AllocatedSize());

            auto texture = device_tensor->device<Texture>();
            const int width = texture->width();
            const int height = texture->height();
            GLenum format = texture->format();
            GLenum type = texture->type();
            // TODO(breakpoint) why DMA is slower than non DMA
            CopyTextureToHost(cpu_tensor->host(), width, height, device_tensor->device<Texture>()->id(),
                    format, type);
        }
    }

    namespace{
        const GLenum kDataType=GL_FLOAT;
        GLenum kInternalFormat = GL_RGBA32F;
        GLenum kFormat = GL_RGBA;
        // Don't need to change this.
        // We want to draw 2 giant triangles that cover the whole screen.
        struct Vertex {
            float x, y;
        };

        static constexpr size_t kNumVertices = 6;

        const char *vertex_shader_text = "#version 300 es\n"
            "in vec2 point; // input to vertex shader\n"
            "void main() {\n"
            "  gl_Position = vec4(point, 0.0, 1.0);\n"
            "}\n";

        const Vertex vertices[kNumVertices] = {
            {-1.f, -1.f},
            {1.0f, -1.f},
            {1.0f, 1.0f},
            {-1.f, -1.f},
            {-1.f, 1.0f},
            {1.0f, 1.0f},
        };
    }

    void Context::Compute(std::initializer_list<size_t> dim_sizes){
        auto ptr = dim_sizes.begin();
        glDispatchCompute(ptr[0], ptr[1], ptr[2]);
    }

    void Context::Reset(){
        frame_buffer_ = CreateFrameBuffer();
        CreateVertexShader();
    }

    Context::Context(Allocator* allocator)
        :allocator_(allocator){
            // max size allowed when using texture
            LOG(INFO)<<"max group invacations: "<<GetMaxTextureSize();
            // prepare framebuffer and vertex shader first
            // as for fragment shader, it is used as compute kernel
            Reset();
        }

    void Context::ConvertTensorHWN4C4ToNCHW(void* src, Tensor* tensor){
        float* nchw_data = tensor->host<float>();
        float* hwn4c4_data = (float*)src;
        const int num_elements = tensor->num_elements();
        const int c = tensor->shape()[1];
        const int h = tensor->shape()[2];
        const int w = tensor->shape()[3];
        const int n = tensor->shape()[0];
        const int up_channel = UP_DIV(c, 4)*4;
        const int n4 = UP_DIV(n, 4);
        const int c4 = UP_DIV(c, 4);

        for(int i=0; i<num_elements; ++i){
            int cur = i;
            const int w_i = cur%w;
            cur/=w;
            const int h_i = cur%h;
            cur/=h;
            const int c_i = cur%c;
            cur/=c;
            const int n_i = cur;
            const int offset = ((((h_i*w+w_i)*n4+n_i/4)*c4+c_i/4)*4+c_i%4)*4+n_i%4;

            nchw_data[i]=hwn4c4_data[offset];
        }
    }

    void Context::ConvertTensorFromStride4(void* src, Tensor* dst_tensor){
        auto shape = dst_tensor->shape();
        const int num_dims = shape.size();
        const int dst_last_dim = shape[num_dims-1];
        const int dst_num_elements = dst_tensor->num_elements();
        const int src_last_dim = UP_ROUND(dst_last_dim, 4);
        float* src_data = (float*)src;
        float* dst_data = dst_tensor->host<float>();
        for(int i=0; i < dst_num_elements; ++i){
            const int src_index = i%dst_last_dim+i/dst_last_dim*src_last_dim;
            dst_data[i] = src_data[src_index];
        }
    }

    void Context::ConvertTensorToStride4(const Tensor* src_tensor, void** out){
        auto shape = src_tensor->shape();
        const int num_dims = shape.size();
        const int src_last_dim = shape[num_dims-1];
        const int dst_last_dim = UP_ROUND(src_last_dim, 4);
        const int src_num_elements = src_tensor->num_elements();
        const int dst_num_elements = src_num_elements / src_last_dim * dst_last_dim;

        float* dst_data = new float[dst_num_elements];
        const float* src_data = src_tensor->host<float>();
        memset(dst_data, 0, sizeof(float)*dst_num_elements);
        // only use data if it is in src
        for(int i=0;i<src_num_elements;++i){
            const int dst_index = i/src_last_dim*dst_last_dim+i%src_last_dim;
            dst_data[dst_index] = src_data[i];
        }

        *out = dst_data;
    }

    void Context::ConvertTensorNCHWToNHWC4(const Tensor* cpu_tensor, void** out){

        const int n = cpu_tensor->shape()[0];
        const int c = cpu_tensor->shape()[1];
        const int h = cpu_tensor->shape()[2];
        const int w = cpu_tensor->shape()[3];

        const int num_elements = n*UP_DIV(c, 4)*4*h*w;
        float* data = new float[num_elements];
        memset(data, 0, sizeof(float)*num_elements);
        float* orig_data = cpu_tensor->host<float>();
        for(int i=0;i<cpu_tensor->num_elements();++i){
            int cur = i;
            const int w_i = cur%w;
            cur/=w;
            const int h_i = cur%h;
            cur/=h;
            const int c_i = cur%c;
            cur/=c;
            const int n_i = cur;
            const int offset = (((n_i*h+h_i)*w+w_i)*UP_DIV(c, 4)+c_i/4)*4+c_i%4;
            data[offset] = orig_data[i];
        }
        *out = data;
    }
    void Context::ConvertTensorNHWCToNHWC4(const Tensor* src_tensor, Tensor* dst_tensor){
        // check format for src_tensor and dst_tensor
        CHECK_EQ(src_tensor->dformat(), ::dlxnet::TensorProto::NHWC);
        CHECK_EQ(dst_tensor->dformat(), ::dlxnet::TensorProto::NHWC4);

        // check memory type
        CHECK(src_tensor->is_host());
        CHECK(dst_tensor->is_host());

        // check size
        CHECK_EQ(src_tensor->RequestedSize(), dst_tensor->RequestedSize());
        const int num_elements = src_tensor->num_elements();

        float* src_data = src_tensor->host<float>();
        const int dst_channel = dst_tensor->channel()*4;
        const int src_channel = src_tensor->channel();
        float* dst_data = dst_tensor->host<float>();
        for(int i=0;i<num_elements;++i){
            if(i%dst_channel<src_channel){
                dst_data[i] = src_data[i/dst_channel*src_channel+i%dst_channel];
            }
        }
    }

    void Context::CopyCPUTensorToDevice(const Tensor* cpu_tensor, Tensor* device_tensor){
        // insanity check first
        // the same data format, nhwc4
        void* data = nullptr;

        if(cpu_tensor->dformat()==dlxnet::TensorProto::ANY
                && device_tensor->dformat()==dlxnet::TensorProto::ANY4){
            // for general tensor case
            // ConvertTensorToStride4(cpu_tensor, &data);
            auto src_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT,
                        cpu_tensor->shape(), Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY));
            Tensor* src_gpu_tensor = src_gpu_tensor_ptr.get();
            internal::CopyCPUTensorToDevice(cpu_tensor, src_gpu_tensor);
            // any to any4 (device->device)
            functor::ConvertTensorANYToANY4()(this, src_gpu_tensor, device_tensor);
            return;
        }else if(device_tensor->dformat()==dlxnet::TensorProto::NHWC4){
            if(cpu_tensor->dformat()== dlxnet::TensorProto::NHWC){
                // convert to nhwc4
                // ConvertTensorNHWCToNHWC4(cpu_tensor, &data);
                // TODO(breakpoint) image is too large for opengl device to use the following code
                // try to fix it
                auto src_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT,
                            cpu_tensor->shape(), Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY));
                Tensor* src_gpu_tensor = src_gpu_tensor_ptr.get();
                internal::CopyCPUTensorToDevice(cpu_tensor, src_gpu_tensor);
                // any to any4 (device->device)
                // here use any -> nhwc4 is the same as nhwc -> nhwc4
                // due to any and nhwc has the same layout in host memory
                functor::ConvertTensorANYToNHWC4()(this, src_gpu_tensor, device_tensor);
                return ;
            }else if(cpu_tensor->dformat()== dlxnet::TensorProto::NCHW){
                LOG(FATAL)<<"Removed now";
                // ConvertTensorNCHWToNHWC4(cpu_tensor, &data);
            }else{
                LOG(FATAL)<<"Unsupported cpu_tensor dformat: "<<cpu_tensor->dformat();
            }
        }else if(device_tensor->dformat()==dlxnet::TensorProto::HWN4C4){
            // CHECK_EQ(cpu_tensor->dformat(), dlxnet::TensorProto::NCHW)
            // <<"dformat of cpu tensor for filter should be NCHW";
            // ConvertTensorNCHWToHWN4C4(cpu_tensor, &data);
            auto src_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT,
                        cpu_tensor->shape(), Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY));
            Tensor* src_gpu_tensor = src_gpu_tensor_ptr.get();
            internal::CopyCPUTensorToDevice(cpu_tensor, src_gpu_tensor);
            functor::ConvertTensorNCHWToHWN4C4()(this, src_gpu_tensor, device_tensor);

            auto dst_cpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(device_tensor));
            Tensor* dst_cpu_tensor = dst_cpu_tensor_ptr.get();
            CopyDeviceTensorToCPU(device_tensor, dst_cpu_tensor);
            return;
        }else{
            data = cpu_tensor->host();
        }

        auto texture = device_tensor->device<Texture>();
        const int width = texture->width();
        const int height = texture->height();
        GLenum format = texture->format();
        GLenum type = texture->type();
        // TODO(breakpoint) why DMA is slower than non DMA
        CopyHostToTexture(data, width, height, device_tensor->device<Texture>()->id(),
                format, type);

        // TODO(breakpoint) use Tensor
        if(data){
            free(data);
        }
    }
    void Context::ConvertTensorNCHWToHWN4C4(const Tensor* tensor, void** out){
        // handle pytorch filter
        // from (out, in, h, w) to (h*w, out_4*in_4*in4, out4)
        // where in_4 = UP_DIV(in, 4), out_4=UP_DIV(out, 4), in4=out4=4
        const int n_out = tensor->shape()[0];
        const int n_in = tensor->shape()[1];
        const int h = tensor->shape()[2];
        const int w = tensor->shape()[3];
        const int in_4 = UP_DIV(n_in, 4);
        const int out_4 = UP_DIV(n_out, 4);
        const int num_elements = h*w*in_4*out_4*4*4;
        float* data = new float[num_elements];
        memset(data, 0, sizeof(float)*num_elements);
        float* orig_data = tensor->host<float>();
        for(int i=0;i<tensor->num_elements();++i){
            // decompose i to four-element tuple
            int cur = i;
            const int w_i = cur%w;
            cur /= w;
            const int h_i = cur%h;
            cur/=h;
            const int in_i = cur%n_in;
            cur/=n_in;
            const int out_i = cur;

            // then compose them to 3-element tuple
            const int hw_i = h_i*w+w_i;
            const int io4_i = (out_i/4 *in_4+in_i/4)*4+in_i%4;
            const int offset = (hw_i*in_4*out_4*4+io4_i)*4+out_i%4;
            data[offset] = orig_data[i];
        }
        *out = data;
    }

    void Context::ConvertTensorNHWCToNHWC4(const Tensor* tensor, void** out){
        // otherwise fall to nhwc4 case
        const int image_height = tensor->num()*tensor->height();
        const int image_width = UP_DIV(tensor->channel(), 4) * tensor->width();
        const int orig_channel = tensor->channel();
        size_t num_elements = image_height * image_width * 4;
        // copy from data to host_
        float* data = new float[num_elements];
        memset(data, 0, sizeof(float)*num_elements);
        float* orig_data = tensor->host<float>();
        const int up_channel = UP_DIV(tensor->channel(), 4)*4;
        for(int i=0;i<num_elements;++i){
            if(i%up_channel<orig_channel){
                data[i] = orig_data[i/up_channel*orig_channel+i%up_channel];
            }
        }
        *out = data;
    }

    void Context::ConvertTensorNHWC4ToNHWC(void* out, Tensor* tensor){
        // tensor->set_host(out);
        float* nhwc_data = tensor->host<float>();
        float* nhwc4_data = (float*)out;
        const int num_elements = tensor->num_elements();
        const int up_channel = UP_DIV(tensor->last_stride(), 4)*4;
        const int channel = tensor->last_stride();

        // there is different in their base number in the last dim(one is channel,
        // the other is up_channel)
        for(int i=0;i<num_elements;++i){
            const int offset = i/channel*up_channel+i%channel;
            nhwc_data[i] = nhwc4_data[offset];
        }
    }




    void Context::CopyDeviceTensorToCPU(const Tensor* device_tensor, Tensor* cpu_tensor){
        float* data = new float[device_tensor->AllocatedSize()/sizeof(float)];
        auto texture = device_tensor->device<Texture>();

        const int width = texture->shape()[0];
        const int height = texture->shape()[1];
        // copy texture to host first

        GLenum format = texture->format();
        GLenum type = texture->type();
        CopyTextureToHost(data, width, height, device_tensor->device<Texture>()->id(),
                format, type);
        if(device_tensor->dformat()==dlxnet::TensorProto::ANY4
                &&cpu_tensor->dformat()==dlxnet::TensorProto::ANY){
            ConvertTensorFromStride4(data, cpu_tensor);
        }else if(device_tensor->dformat()==dlxnet::TensorProto::NHWC4){
            // only one cpu dformat supported here
            // may be more target dformat will be supported in the future
            CHECK(internal::IsCPUDFormat(cpu_tensor->dformat()));
            // copy data to cpu_tensor
            ConvertTensorNHWC4ToNHWC(data, cpu_tensor);
        }else if(device_tensor->dformat()==dlxnet::TensorProto::HWN4C4){
            // now used to download filter generated from onnx model
            ConvertTensorHWN4C4ToNCHW(data, cpu_tensor);
        }else{
            LOG(FATAL)<<"unsupported conversion from device_dformat: "
                <<device_tensor->dformat()<<" -> cpu_dformat: "
                <<cpu_tensor->dformat();
        }

        //TODO(breakpoint) cache it
        free(data);
    }

    void Context::CopyImageToBuffer(Texture* texture, Buffer* buffer){
        // program
        Program program = Program();
        program.AttachFile(kImage2buffer_name_).Link();

        GLuint ray_program = program.program_id();
        // GLuint tex_output = texture->id();
        // GLuint SSBO = buffer->id();

        int tex_w = texture->shape()[0];
        int tex_h = texture->shape()[1];

        program.Activate();

        //set param in program and then dispatch the shaders
        {
            // set input and output
            program.set_vec2i("image_shape", tex_w, tex_h);
            OPENGL_CHECK_ERROR;
            program.set_input_sampler2D(texture->id(), texture->format());
            OPENGL_CHECK_ERROR;
            program.set_buffer(buffer->id(), buffer->target());
            OPENGL_CHECK_ERROR;

            glDispatchCompute((GLuint)tex_w, (GLuint)tex_h, 1);
            OPENGL_CHECK_ERROR;
        }

        // sync
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    void Context::CopyBufferToImage(Texture* texture, Buffer* buffer){
        //(TODO cache program if possible)
        // program
        Program program = Program();
        program.AttachFile(kBuffer2image_name_).Link();


        GLuint ray_program = program.program_id();

        int tex_w = texture->shape()[0];
        int tex_h = texture->shape()[1];

        program.Activate();

        //set param in program and then dispatch the shaders
        {
            // set input and output
            program.set_vec2i("image_shape", tex_w, tex_h);
            OPENGL_CHECK_ERROR;
            program.set_output_sampler2D(texture->id(), texture->format());
            OPENGL_CHECK_ERROR;
            program.set_buffer(buffer->id(), buffer->target());
            OPENGL_CHECK_ERROR;

            glDispatchCompute((GLuint)tex_w, (GLuint)tex_h, 1);
        }

        // sync
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    void Context::CopyCPUBufferToDevice(Buffer* buffer, void* buffer_cpu){
        // upload
        auto ptr = buffer->Map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
        ::memcpy(ptr, buffer_cpu, buffer->size());
        buffer->UnMap();
    }

    void Context::CopyDeviceBufferToCPU(Buffer* buffer, void* buffer_cpu){
        // download
        auto ptr = buffer->Map(GL_MAP_READ_BIT);
        ::memcpy(buffer_cpu, ptr, buffer->size());
        buffer->UnMap();
    }

    void Context::CreateVertexShader(){
        // We always render the same vertices and triangles.
        GLuint vertex_buffer;
        OPENGL_CALL(glGenBuffers(1, &vertex_buffer));
        OPENGL_CALL(glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer));
        OPENGL_CALL(glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices,
                    GL_STATIC_DRAW));

        GLuint vertex_array;
        OPENGL_CALL(glGenVertexArrays(1, &vertex_array));
        OPENGL_CALL(glBindVertexArray(vertex_array));
        OPENGL_CALL(glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer));

        // We always use the same vertex shader.
        vertex_shader_ = CreateShader(GL_VERTEX_SHADER, vertex_shader_text);
    }

    Context::~Context(){
        OPENGL_CALL(glDeleteFramebuffers(1, &frame_buffer_));
    }

    Program* Context::CreateProgram(const std::string& kernel_fname){
        if(kernel_fname.empty()){
            // no kernel program needed for this op, like const op
            return nullptr;
        }
        // set program
        // program_ .reset(new Program);
        auto program = new Program;
        (*program).AttachFile(kernel_fname, GL_FRAGMENT_SHADER)
            .AttachShader(vertex_shader_);
        program->Link();
        program->Activate();
        // set vertex shader first
        // then you can set fragment shader to do actually computation
        program->SetVertexShader();
        return program;
    }


    Context* GetContext(){
        static Context* context = new Context;
        return context;
    }
}//namespace opengl
