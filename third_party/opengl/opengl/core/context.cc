#include <cstring>
#include <glog/logging.h>

#include "opengl/core/context.h"
#include "opengl/core/program.h"
#include "opengl/utils/macros.h"


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
            // work group size
            int work_grp_cnt[3];
            //OPENGL_CALL(glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &work_grp_cnt[0]));
            //OPENGL_CALL(glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &work_grp_cnt[1]));
            //OPENGL_CALL(glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &work_grp_cnt[2]));
            LOG(INFO)<<"max global (total) work group counts x:"<< work_grp_cnt[0]
                <<" y:"<<work_grp_cnt[1]
                <<" z:"<< work_grp_cnt[2];
            int work_grp_inv;
            // OPENGL_CALL(glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &work_grp_inv));
            LOG(INFO)<<"max group invacations: "<<work_grp_inv;
        }

    void Context::CopyCPUTensorToDevice(const Tensor* cpu_tensor, Tensor* device_tensor){
        // insanity check first
        // the same data format, nhwc4
        void* nhwc4_data=nullptr;
        if(cpu_tensor->dformat()==dlxnet::TensorProto::NHWC){
            // convert to nhwc4
            ConvertTensorNHWCToNHWC4(cpu_tensor, &nhwc4_data);
        }else{
            nhwc4_data = cpu_tensor->host();
        }
        // same number of bytes
        // CHECK_EQ(cpu_tensor->size(), device_tensor->size());
        const int width = device_tensor->device<Texture>()->shape()[0];
        const int height = device_tensor->device<Texture>()->shape()[1];
        OPENGL_CALL(glBindTexture(GL_TEXTURE_2D, device_tensor->device<Texture>()->id()));
        OPENGL_CALL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                    kFormat, kDataType, nhwc4_data));
    }

    void Context::ConvertTensorNHWCToNHWC4(const Tensor* tensor, void** out){
        // make sure it is in cpu host
        CHECK_EQ(tensor->mem_type(), Tensor::HOST_MEMORY);

        // otherwise fall to nhwc4 case
        const int image_height = tensor->num()*tensor->height();
        const int image_width = UP_DIV(tensor->channel(), 4) * tensor->width();
        const int orig_channel = tensor->channel();
        size_t num_elements = image_height * image_width * 4;
        size_t bytes = num_elements * sizeof(float);
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
        // insanity check first
        // the same data format, nhwc4
        // CHECK_EQ(cpu_tensor->dformat(), device_tensor->dformat());
        // same number of bytes
        // CHECK_EQ(cpu_tensor->size(), device_tensor->size());
        void* nhwc4_data=nullptr;
        if(cpu_tensor->dformat()==dlxnet::TensorProto::NHWC){
            // convert to nhwc4
            ConvertTensorNHWCToNHWC4(cpu_tensor, &nhwc4_data);
        }else{
            nhwc4_data = cpu_tensor->host();
        }

        GLint ext_format, ext_type;
        const int width = device_tensor->device<Texture>()->shape()[0];
        const int height = device_tensor->device<Texture>()->shape()[1];

        glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &ext_format);
        glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &ext_type);
        CHECK_EQ(ext_type, kDataType)<<"unmatched type";
        CHECK_EQ(ext_format, kFormat)<<"unmatched format";

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                device_tensor->device<Texture>()->id() , 0);
        // download
        OPENGL_CALL(glReadBuffer(GL_COLOR_ATTACHMENT0));
        OPENGL_CALL(glReadPixels(0, 0, width, height, ext_format, ext_type, nhwc4_data));

        // copy nhwc4_data to cpu_tensor
        ConvertTensorNHWC4ToNHWC(nhwc4_data, cpu_tensor);
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
