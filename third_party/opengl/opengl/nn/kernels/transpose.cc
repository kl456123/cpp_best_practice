#include "opengl/nn/kernels/transpose.h"

#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"


namespace opengl{
    TransposeKernel::TransposeKernel(Context* context)
        :Kernel(context){
            kernel_fname_ = "../opengl/nn/glsl/transpose.glsl";
        }

    void TransposeKernel::SetupAttr(const dlxnet::Attribute& attr){
        auto& transpose_params = attr.transpose_attr();
        for(auto item: transpose_params.perm()){
            perm_.emplace_back(item);
        }
    }

    void TransposeKernel::Compute(TensorList& inputs, TensorList& outputs){
        DLOG(INFO)<<"TransposeKernel Inputs: "<<inputs.size();
        program_->Activate();
        auto input_image = inputs[0]->device<Texture>();

        SetFrameBuffer(outputs);
        SetVertexShader();
        program_->set_vec4i("perm", perm_);

        // input
        {
            program_->set_image2D("input_image", input_image->id(),  0);
            OPENGL_CHECK_ERROR;
        }

        OPENGL_CALL(glClear(GL_COLOR_BUFFER_BIT));
        OPENGL_CALL(glDrawArrays(GL_TRIANGLES, 0, 6));
        glFinish();
    }

    void TransposeKernel::InferOutputShape(TensorShapeList& input_shapes,
            TensorShapeList& output_shapes){
        // set output dformat first, then we can according
        // to dformat to infer output shape
        output_tensor_dformats_.emplace_back(dlxnet::TensorProto::NHWC4);

        output_shapes.clear();
        output_shapes.resize(1);
        CHECK_EQ(input_shapes.size(), 1);
        output_shapes[0] = input_shapes[0];
    }

    TransposeKernel::~TransposeKernel(){}

    REGISTER_KERNEL_WITH_NAME(TransposeKernel, "Transpose");
}//namespace opengl
