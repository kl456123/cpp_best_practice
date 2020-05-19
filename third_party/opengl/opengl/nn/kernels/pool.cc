#include "opengl/nn/kernels/pool.h"

#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"


namespace opengl{
    PoolKernel::PoolKernel(Context* context)
        :Kernel(context){
            kernel_fname_ = "../opengl/nn/glsl/pool.glsl";
        }

    void PoolKernel::SetupAttr(const dlxnet::Attribute& attr){
        auto& maxpool_params = attr.maxpool_attr();

        // single output
        output_tensor_dformats_.emplace_back(dlxnet::TensorProto::NHWC4);

        // handle pads
        CHECK_EQ(maxpool_params.pads_size(), 4);
        for(auto& pad: maxpool_params.pads()){
            CHECK_EQ(maxpool_params.pads(0),pad);
        }
        padding_ = maxpool_params.pads(0);

        // handle stride
        CHECK_EQ(maxpool_params.strides_size(), 2);
        CHECK_EQ(maxpool_params.strides(0), maxpool_params.strides(1));
        stride_ = maxpool_params.strides(0);


        // handle kernel
        CHECK_EQ(maxpool_params.kernel_shape_size(), 2);
        CHECK_EQ(maxpool_params.kernel_shape(0), maxpool_params.kernel_shape(1));
        kernel_size_=maxpool_params.kernel_shape(0);
    }

    void PoolKernel::Compute(TensorList& inputs, TensorList& outputs){
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

    void PoolKernel::InferOutputShape(TensorShapeList& input_shapes,
            TensorShapeList& output_shapes){
        CHECK_EQ(input_shapes.size(), 1);
        output_shapes.clear();
        output_shapes.resize(1);

        // compute output shape like conv2d
        output_shapes[0] = input_shapes[0];
    }

    PoolKernel::~PoolKernel(){}

    REGISTER_KERNEL_WITH_NAME(PoolKernel, "MaxPool");
    // REGISTER_KERNEL_WITH_NAME(PoolKernel, "GlobalAveragePool");
}//namespace opengl
