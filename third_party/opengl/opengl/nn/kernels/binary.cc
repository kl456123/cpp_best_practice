#include "opengl/nn/kernels/binary.h"

#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"



namespace opengl{
    BinaryKernel::BinaryKernel(Context* context)
        :Kernel(context){
            // set work size
            for(int i=0;i<3;i++){
                work_sizes_[i] = 1;
            }

            // set program
            program_ = new Program;
            std::string fname = "../opengl/examples/gpgpu/bias_add.glsl";
            program_->AttachFile(fname);
            program_->Link();
        }


    BinaryKernel::~BinaryKernel(){
        if(program_!=nullptr){delete program_;}
    }

    void BinaryKernel::Compute(TensorList& inputs, TensorList& outputs){
        auto texture1 = inputs[0]->device<Texture>();
        auto texture2 = inputs[1]->device<Texture>();
        auto texture3 = outputs[0]->device<Texture>();

        program_->Activate();
        int tex_w = texture1->shape()[0];
        int tex_h = texture1->shape()[1];

        program_->set_vec2i("image_shape", tex_w, tex_h);
        glBindImageTexture(0, texture3->id(), 0, GL_TRUE, 0, GL_WRITE_ONLY, texture3->format());
        OPENGL_CHECK_ERROR;
        // input0
        {
            program_->set_image2D("input0", texture1->id(),  0);
            OPENGL_CHECK_ERROR;
        }

        // input1
        {
            program_->set_image2D("input1", texture2->id(),  1);
            OPENGL_CHECK_ERROR;
        }
        glDispatchCompute((GLuint)tex_w, (GLuint)tex_h, 1);

    }

    void BinaryKernel::InferOutputShape(TensorShapeList& input_shapes,
            TensorShapeList& output_shapes){
        output_shapes.clear();
        output_shapes.resize(1);
        for(auto& input_shape:input_shapes){
            // check input is the same shape
        }
        output_shapes[0] = input_shapes[0];
    }

    REGISTER_KERNEL_WITH_NAME(BinaryKernel, "Add");
}//namespace opengl



