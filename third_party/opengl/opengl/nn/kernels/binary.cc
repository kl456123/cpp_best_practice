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
            kernel_fname_ = "../opengl/nn/glsl/binary.glsl";
        }


    BinaryKernel::~BinaryKernel(){
        if(program_!=nullptr){delete program_;}
    }

    void BinaryKernel::Compute(TensorList& inputs, TensorList& outputs){
        OPENGL_CALL(glUseProgram(program_->program_id()));
        auto texture1 = inputs[0]->device<Texture>();
        auto texture2 = inputs[1]->device<Texture>();
        SetFrameBuffer(outputs);
        SetVertexShader();


        program_->Activate();
        int tex_w = texture1->shape()[0];
        int tex_h = texture1->shape()[1];

        program_->set_vec2i("image_shape", tex_w, tex_h);
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

        OPENGL_CALL(glClear(GL_COLOR_BUFFER_BIT));
        OPENGL_CALL(glDrawArrays(GL_TRIANGLES, 0, 6));
        glFinish();
        // glDispatchCompute((GLuint)tex_w, (GLuint)tex_h, 1);

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

    void BinaryKernel::SetupAttr(const dlxnet::Attribute& attr){}

    REGISTER_KERNEL_WITH_NAME(BinaryKernel, "Add");
}//namespace opengl



