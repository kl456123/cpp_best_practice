#include <cstring>
#include <glog/logging.h>

#include "opengl/core/context.h"
#include "opengl/core/program.h"
#include "opengl/utils/macros.h"


void Context::Compute(std::initializer_list<size_t> dim_sizes){
    auto ptr = dim_sizes.begin();
    glDispatchCompute(ptr[0], ptr[1], ptr[2]);
}

Context::Context(Allocator* allocator)
    :allocator_(allocator){

        // work group size
        int work_grp_cnt[3];
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &work_grp_cnt[0]);
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &work_grp_cnt[1]);
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &work_grp_cnt[2]);
        LOG(INFO)<<"max global (total) work group counts x:"<< work_grp_cnt[0]
            <<" y:"<<work_grp_cnt[1]
            <<" z:"<< work_grp_cnt[2];
        int work_grp_inv;
        glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &work_grp_inv);
        LOG(INFO)<<"max group invacations: "<<work_grp_inv;
    }

void Context::CopyCPUTensorToDevice(const Tensor* cpu_tensor, Tensor* device_tensor){
    // do some necessary check first

    // check cpu_tensor is in cpu
    CHECK(cpu_tensor->is_host());

    // check same size
    CHECK(cpu_tensor->size()==device_tensor->size());
    CHECK(cpu_tensor->num_elements()==device_tensor->num_elements());

    Buffer* device_buffer = reinterpret_cast<Buffer*>(device_tensor->device());
    ::memcpy(device_buffer->Map(GL_MAP_WRITE_BIT),cpu_tensor->host(),
            cpu_tensor->size());

    device_buffer->UnMap();
}


void Context::CopyDeviceTensorToCPU(const Tensor* device_tensor, Tensor* cpu_tensor){
    // do some necessary check first

    // check cpu_tensor is in cpu
    CHECK(cpu_tensor->is_host());

    // check same size
    CHECK(cpu_tensor->num_elements()==device_tensor->num_elements());

    Buffer* device_buffer = reinterpret_cast<Buffer*>(device_tensor->device());
    ::memcpy(cpu_tensor->host(), device_buffer->Map(GL_MAP_READ_BIT),
            cpu_tensor->size());

    device_buffer->UnMap();
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
        program.set_buffer(buffer->id(), buffer->target())
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
        program.set_buffer(buffer->id(), buffer->target())
        OPENGL_CHECK_ERROR;

        glDispatchCompute((GLuint)tex_w, (GLuint)tex_h, 1);
    }

    // sync
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void Context::CopyCPUBufferToDevice(Buffer* buffer, float* buffer_cpu){
    // upload
    auto ptr = buffer->Map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    ::memcpy(ptr, buffer_cpu, buffer->size());
    buffer->UnMap();
}

void Context::CopyDeviceBufferToCPU(Buffer* buffer, float* buffer_cpu){
    // download
    auto ptr = buffer->Map(GL_MAP_READ_BIT);
    ::memcpy(buffer_cpu, ptr, buffer->size());
    buffer->UnMap();
}
