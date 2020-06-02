#include "opengl/nn/kernels/reshape.h"
#include "opengl/core/fbo_session.h"

#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"


namespace opengl{
    ReshapeKernel::ReshapeKernel(Context* context)
        :Kernel(context){
            kernel_fname_ = "../opengl/nn/glsl/reshape.glsl";
        }

    void ReshapeKernel::SetupAttr(const dlxnet::Attribute& attr){
        output_tensor_dformats_.emplace_back(dlxnet::TensorProto::NHWC4);
    }

    void ReshapeKernel::Compute(TensorList& inputs, TensorList& outputs){
        DLOG(INFO)<<"ReshapeKernel Inputs: "<<inputs.size();
        program_->Activate();
        auto input_image = inputs[0]->device<Texture>();

        SetFrameBuffer(outputs);
        SetVertexShader();

        // input
        {
            program_->set_image2D("input_image", input_image->id(),  0);
            OPENGL_CHECK_ERROR;
        }

        OPENGL_CALL(glClear(GL_COLOR_BUFFER_BIT));
        OPENGL_CALL(glDrawArrays(GL_TRIANGLES, 0, 6));
        glFinish();
    }

    void ReshapeKernel::InferOutputShape(const TensorList& inputs,
            TensorShapeList& output_shapes){
        output_shapes.clear();
        output_shapes.resize(1);
        CHECK_EQ(inputs.size(), 2);
        // make sure it is prepared(computed)
        auto shape_tensor = inputs[1];

        Tensor* cpu_tensor = new Tensor(Tensor::DT_FLOAT, shape_tensor->shape(),
                Tensor::HOST_MEMORY, dlxnet::TensorProto::NHWC);
        session_->context()->CopyDeviceTensorToCPU(shape_tensor, cpu_tensor);
        for(int i=0;i<cpu_tensor->num_elements();++i){
            output_shapes[0].emplace_back(cpu_tensor->host<float>()[i]);
        }

        delete cpu_tensor;
    }


    ReshapeKernel::~ReshapeKernel(){}

    REGISTER_KERNEL_WITH_NAME(ReshapeKernel, "Reshape");
}//namespace opengl

