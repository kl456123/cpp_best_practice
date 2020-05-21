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


    BinaryKernel::~BinaryKernel(){}

    void BinaryKernel::Compute(TensorList& inputs, TensorList& outputs){
        program_->Activate();
        auto input0 = inputs[0]->device<Texture>();
        auto input1 = inputs[1]->device<Texture>();
        SetFrameBuffer(outputs);
        SetVertexShader();


        auto input_shape = inputs[0]->shape();
        auto output_shape = outputs[0]->shape();

        program_->set_vec3i("input_shape", inputs[0]->height(),
                inputs[0]->width(), inputs[0]->channel());
        // input
        {
            program_->set_image2D("input0", input0->id(),  0);
            OPENGL_CHECK_ERROR;
        }

        // filter
        {
            program_->set_image2D("input1", input1->id(),  1);
            OPENGL_CHECK_ERROR;
        }

        OPENGL_CALL(glClear(GL_COLOR_BUFFER_BIT));
        OPENGL_CALL(glDrawArrays(GL_TRIANGLES, 0, 6));
        glFinish();
    }

    void BinaryKernel::InferOutputShape(const TensorList& input_tensors,
            TensorShapeList& output_shapes){
        output_shapes.clear();
        output_shapes.emplace_back(input_tensors[0]->shape());
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

    void BinaryKernel::SetupAttr(const dlxnet::Attribute& attr){
        output_tensor_dformats_.emplace_back(dlxnet::TensorProto::NHWC4);
    }

    REGISTER_KERNEL_WITH_NAME(BinaryKernel, "Add");
}//namespace opengl



