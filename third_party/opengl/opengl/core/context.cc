#include <cstring>
#include <glog/logging.h>

#include "opengl/core/context.h"
#include "opengl/core/program.h"
#include "opengl/utils/macros.h"
#include "opengl/core/driver.h"
#include "opengl/core/tensor_format.h"


namespace opengl{
    namespace{
        const GLenum kDataType=GL_FLOAT;
        GLenum kInternalFormat = GL_RGBA32F;
        GLenum kFormat = GL_RGBA;
    }

    void Context::Compute(std::initializer_list<size_t> dim_sizes){
        auto ptr = dim_sizes.begin();
        glDispatchCompute(ptr[0], ptr[1], ptr[2]);
    }

    Context::Context(Allocator* allocator)
        :allocator_(allocator){
            // max size allowed when using texture
            int work_grp_inv;
            OPENGL_CALL(glGetIntegerv(GL_MAX_TEXTURE_SIZE, &work_grp_inv));
            LOG(INFO)<<"max group invacations: "<<work_grp_inv;
        }

    void Context::ConvertTensorHWN4C4ToNCHW(void* src, Tensor* tensor){
        float* nchw_data = tensor->host<float>();
        float* hwn4c4_data = (float*)src;
        const int num_elements = tensor->num_elements();
        const int up_channel = UP_DIV(tensor->channel(), 4)*4;
        const int c = tensor->channel();
        const int h = tensor->height();
        const int w = tensor->width();
        const int n = tensor->num();
        const int n4 = UP_DIV(n, 4);
        const int c4 = UP_DIV(c, 4);

        for(int i=0;i<num_elements;++i){
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
        CHECK_EQ(src_tensor->size(), dst_tensor->size());
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
        // auto shapes = TensorShapeFromFormatv1(device_tensor->dformat(),
                // cpu_tensor->shape(), cpu_tensor->dformat());
        // // // allocate tmp host tensor
        // Tensor host_tensor(device_tensor->dtype(), shapes, Tensor::HOST_MEMORY,
                // device_tensor->dformat());

        // data = host_tensor.host();

        if(device_tensor->dformat()==dlxnet::TensorProto::NHWC4){
            if(cpu_tensor->dformat()== dlxnet::TensorProto::NHWC){
                // convert to nhwc4
                ConvertTensorNHWCToNHWC4(cpu_tensor, &data);
            }else if(cpu_tensor->dformat()== dlxnet::TensorProto::NCHW){
                ConvertTensorNCHWToNHWC4(cpu_tensor, &data);
            }else{
                LOG(FATAL)<<"Unsupported cpu_tensor dformat: "<<cpu_tensor->dformat();
            }
        }else if(device_tensor->dformat()==dlxnet::TensorProto::HWN4C4){
            CHECK_EQ(cpu_tensor->dformat(), dlxnet::TensorProto::NCHW)
                <<"dformat of cpu tensor for filter should be NCHW";
            ConvertTensorNCHWToHWN4C4(cpu_tensor, &data);
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
        const int up_channel = UP_DIV(tensor->channel(), 4)*4;
        const int channel = tensor->channel();

        // there is different in their base number in the last dim(one is channel,
        // the other is up_channel)
        for(int i=0;i<num_elements;++i){
            const int offset = i/channel*up_channel+i%channel;
            nhwc_data[i] = nhwc4_data[offset];
        }
    }


    void Context::CopyDeviceTensorToCPU(const Tensor* device_tensor, Tensor* cpu_tensor){
        float* data = new float[device_tensor->size()/sizeof(float)];
        auto texture = device_tensor->device<Texture>();

        const int width = texture->shape()[0];
        const int height = texture->shape()[1];
        // copy texture to host first

        GLenum format = texture->format();
        GLenum type = texture->type();
        CopyTextureToHost(data, width, height, device_tensor->device<Texture>()->id(),
                format, type);

        if(device_tensor->dformat()==dlxnet::TensorProto::NHWC4){
            // only one cpu dformat supported here
            // may be more target dformat will be supported in the future
            CHECK_EQ(cpu_tensor->dformat(), dlxnet::TensorProto::NHWC);
            // copy data to cpu_tensor
            ConvertTensorNHWC4ToNHWC(data, cpu_tensor);
        }else if(device_tensor->dformat()==dlxnet::TensorProto::HWN4C4){
            // now used to download filter generated from onnx model
            CHECK_EQ(cpu_tensor->dformat(), dlxnet::TensorProto::NCHW);
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

    Context* GetContext(){
        static Context* context = new Context;
        return context;
    }
}//namespace opengl
