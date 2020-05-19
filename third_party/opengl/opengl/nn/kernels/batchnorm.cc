#include "opengl/nn/kernels/batchnorm.h"

#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"


namespace opengl{
    BatchNormKernel::BatchNormKernel(Context* context)
        :Kernel(context){
            kernel_fname_ = "../opengl/nn/glsl/batchnorm.glsl";
        }

    void BatchNormKernel::SetupAttr(const dlxnet::Attribute& attr){
        auto& bn_params = attr.batchnorm_attr();
        momentum_ = bn_params.momentum();
        eps_ = bn_params.epsilon();

        // single output
        output_tensor_dformats_.emplace_back(dlxnet::TensorProto::NHWC4);
    }

    void BatchNormKernel::Compute(TensorList& inputs, TensorList& outputs){
        DLOG(INFO)<<"BatchNormalization Inputs: "<<inputs.size();
        program_->Activate();
        auto input_image = inputs[0]->device<Texture>();
        auto gamma = inputs[1]->device<Texture>();
        auto beta = inputs[2]->device<Texture>();
        auto mean = inputs[3]->device<Texture>();
        auto var = inputs[4]->device<Texture>();

        SetFrameBuffer(outputs);
        SetVertexShader();
        program_->set_float("eps", eps_);
        program_->set_float("momentum", momentum_);
        program_->set_vec3i("output_shape", outputs[0]->height(),
                outputs[0]->width(), outputs[0]->channel());

        // input
        {
            program_->set_image2D("input_image", input_image->id(),  0);
            OPENGL_CHECK_ERROR;
        }

        // gamma
        {
            program_->set_image2D("input_gamma", gamma->id(), 1);
            OPENGL_CHECK_ERROR;
        }

        // beta
        {
            program_->set_image2D("input_beta", beta->id(), 2);
            OPENGL_CHECK_ERROR;
        }

        // mean
        {
            program_->set_image2D("input_mean", mean->id(), 3);
            OPENGL_CHECK_ERROR;
        }

        // variance
        {
            program_->set_image2D("input_var", var->id(), 4);
            OPENGL_CHECK_ERROR;
        }

        OPENGL_CALL(glClear(GL_COLOR_BUFFER_BIT));
        OPENGL_CALL(glDrawArrays(GL_TRIANGLES, 0, 6));
        glFinish();
    }

    void BatchNormKernel::InferOutputShape(TensorShapeList& input_shapes,
            TensorShapeList& output_shapes){
        CHECK_EQ(input_shapes.size(), 5);
        output_shapes.clear();
        output_shapes.resize(1);

        output_shapes[0] = input_shapes[0];
        // input, running_mean, running_var, weight, bias
        // const int channel = input_shapes[0];
        // for(int i=1;i<input_shapes.size();++i){
            // CHECK_EQ(input_shapes[i]->channel(), channel);
        // }
    }

    BatchNormKernel::~BatchNormKernel(){}

    REGISTER_KERNEL_WITH_NAME(BatchNormKernel, "BatchNormalization");
}//namespace opengl
