#include "opengl/nn/kernels/gemm.h"

#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"


namespace opengl{
    GemmKernel::GemmKernel(Context* context)
        :Kernel(context){
            kernel_fname_ = "../opengl/nn/glsl/flatten.glsl";
        }

    void GemmKernel::SetupAttr(const dlxnet::Attribute& attr){
        auto& gemm_params = attr.gemm_attr();
        alpha_ = gemm_params.alpha();
        beta_ = gemm_params.beta();
        transB_ = gemm_params.transb();

        // single output
        output_tensor_dformats_.emplace_back(dlxnet::TensorProto::NHWC4);
    }

    void GemmKernel::Compute(TensorList& inputs, TensorList& outputs){
        DLOG(INFO)<<"GemmKernel Inputs: "<<inputs.size();
        program_->Activate();
        auto input_image = inputs[0]->device<Texture>();

        SetFrameBuffer(outputs);
        SetVertexShader();
        program_->set_vec3i("output_shape", outputs[0]->height(),
                outputs[0]->width(), outputs[0]->channel());

        // input
        {
            program_->set_image2D("input_image", input_image->id(),  0);
            OPENGL_CHECK_ERROR;
        }

        OPENGL_CALL(glClear(GL_COLOR_BUFFER_BIT));
        OPENGL_CALL(glDrawArrays(GL_TRIANGLES, 0, 6));
        glFinish();
    }

    void GemmKernel::InferOutputShape(TensorShapeList& input_shapes,
            TensorShapeList& output_shapes){
        // Y = WX+B
        // X, W, B, like conv2d
        // X: (N, 1, 1, C_in)
        // W: (C_out, C_in, 1, 1)
        // B: (1, C_out, 1, 1)
        // Y: (N, 1, 1, C_out)
        // TODO(breakpoint) merge gemm to conv1x1
        CHECK_EQ(input_shapes.size(), 3);
        output_shapes.clear();
        output_shapes.resize(1);
        // check in_channels
        CHECK_EQ(input_shapes[1][1], input_shapes[0][3]);
        // check out_channels
        // TODO(breakpoint) bias shape is not correct, it should be NHWC instead of NCHW
        CHECK_EQ(input_shapes[1][0], input_shapes[2][1]);

        // Y: (N, 1, 1, C_out)
        output_shapes[0]={input_shapes[0][0], 1, 1, input_shapes[1][0]};
    }

    GemmKernel::~GemmKernel(){}

    REGISTER_KERNEL_WITH_NAME(GemmKernel, "Gemm");
}//namespace opengl
