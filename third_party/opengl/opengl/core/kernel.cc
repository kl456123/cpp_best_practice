#include "opengl/core/kernel.h"
#include "opengl/core/program.h"
#include "opengl/utils/macros.h"


namespace opengl{
    namespace{
        struct Vertex {
            float x, y;
        };
    }
    Kernel::Kernel(Context* context)
        :context_(context){}

    Kernel::~Kernel(){}

    void Kernel::SetupProgram(GLuint vertex_shader){
        if(kernel_fname_.empty()){
            // no kernel program needed for this op, like const op
            return;
        }
        // set program
        program_ = new Program;
        (*program_).AttachFile(kernel_fname_, GL_FRAGMENT_SHADER)
            .AttachShader(vertex_shader);
        program_->Link();
    }

    void Kernel::SetVertexShader(){
        // set input arguments for vertex shader
        auto point_attrib = GLuint(glGetAttribLocation(program_->program_id(), "point"));
        OPENGL_CALL(glEnableVertexAttribArray(point_attrib));
        OPENGL_CALL(glVertexAttribPointer(point_attrib, 2, GL_FLOAT, GL_FALSE,
                    sizeof(Vertex), nullptr));
    }

    void Kernel::SetFrameBuffer(TensorList& outputs){
        CHECK_EQ(outputs.size(), 1);
        CHECK_EQ(outputs[0]->mem_type(), Tensor::DEVICE_TEXTURE);

        const int width = outputs[0]->shape()[0];
        const int height = outputs[0]->shape()[1];
        // OPENGL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_));
        OPENGL_CALL(glViewport(0, 0, width, height));


        auto output_texture = outputs[0]->device<Texture>()->id();

        // Set "renderedTexture" as our colour attachement #0
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,
                output_texture , 0);

        // Set the list of draw buffers.
        // GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
        // "1" is the size of DrawBuffers.
        // OPENGL_CALL(glDrawBuffers(1, DrawBuffers));

        // Always check that our framebuffer is ok
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            LOG(FATAL) << "Framebuffer not complete.";
        }
    }


}//namespace opengl
