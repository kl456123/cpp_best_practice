#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <memory>
#include <string.h>
#include <glog/logging.h>

#include "opengl/core/init.h"
#include "opengl/core/texture.h"
#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/core/buffer.h"
#include "opengl/utils/macros.h"

// using namespace opengl;
using opengl::Texture;
using opengl::Context;
using opengl::Buffer;
using opengl::Program;
using opengl::ShaderBuffer;



void BiasAdd(Texture* texture1, Texture* texture2, Texture* texture3){
    auto program = std::unique_ptr<Program>(new Program);
    program->AttachFile("../opengl/examples/gpgpu/bias_add.glsl");
    program->Link();
    program->Activate();
    int tex_w = texture1->shape()[0];
    int tex_h = texture1->shape()[1];

    program->set_vec2i("image_shape", tex_w, tex_h);
    glBindImageTexture(0, texture3->id(), 0, GL_TRUE, 0, GL_WRITE_ONLY, texture3->format());
    OPENGL_CHECK_ERROR;
    // input0
    {
        program->set_image2D("input0", texture1->id(),  0);
        OPENGL_CHECK_ERROR;
    }

    // input1
    {
        program->set_image2D("input1", texture2->id(),  1);
        OPENGL_CHECK_ERROR;
    }
    glDispatchCompute((GLuint)tex_w, (GLuint)tex_h, 1);
}

int main(int arc, char* argv[]){
    google::InitGoogleLogging(argv[0]);
    // init window and context
#ifdef ARM_PLATFORM
    ::opengl::egl_init();
#else
    ::opengl::glfw_init(1280, 800);
#endif


    // init glew
    ::opengl::glew_init();

    auto gl_context = std::unique_ptr<Context>(new Context);

    //////////////////////////////////////////////
    // parameters
    // Creating the Texture / Image
    // dimensions of the image
    // just use single channel(R in RGBA)
    // shape
    typedef float DataType;
    int tex_w = 512, tex_h = 512;
    GLenum internal_format= GL_RGBA32F;
    GLenum target = GL_TEXTURE_2D;
    GLenum format = GL_RGBA;
    GLenum type = GL_FLOAT;
    const int channels = 4;
    std::vector<int> image_shape({tex_h, tex_w, channels});
    // num of elements
    int num = 1;
    for(auto dim:image_shape){
        num*=dim;
    }
    // size
    const int size = num * sizeof(DataType);

    //////////////////////////////////////////////
    // cpu data
    DataType* image_data = new DataType[num];
    for(int i=0;i<num;i++){
        image_data[i] = random()%256/256.0;
    }
    DataType *buffer_cpu = new DataType[num];
    memset(buffer_cpu, 0, size);

    //////////////////////////////////////////////
    // gpu data , all is empty
    // texture and buffer
    auto texture1 = std::unique_ptr<Texture>(
            new Texture(image_shape, internal_format, target, image_data));
    auto texture2 = std::unique_ptr<Texture>(
            new Texture(image_shape, internal_format, target, image_data));
    auto texture3 = std::unique_ptr<Texture>(
            new Texture(image_shape, internal_format, target));

    auto buffer = std::unique_ptr<Buffer>(new ShaderBuffer(size));
    ///////////////////////////////////////////////

    // upload
    // gl_context->CopyCPUBufferToDevice(buffer.get(), image_data);

    // reshape
    // gl_context->CopyBufferToImage(texture1.get(), buffer.get());

    // calculate by using texture
    BiasAdd(texture1.get(), texture2.get(), texture3.get());

    // flatten
    gl_context->CopyImageToBuffer(texture3.get(), buffer.get());

    // download
    gl_context->CopyDeviceBufferToCPU(buffer.get(), buffer_cpu);

    ///////////////////////////////////////////
    // just print
    for(int i=0;i<100;i++){
        std::cout<<(DataType)image_data[i]<<" ";
    }

    std::cout<<std::endl;
    std::cout<<std::endl;
    for(int i=0;i<100;i++){
        std::cout<<(DataType)buffer_cpu[i]<<" ";
    }
    std::cout<<std::endl;

    return 0;
}
