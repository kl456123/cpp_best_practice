#include "opengl/nn/kernels/unsqueeze.h"

#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"


namespace opengl{
    UnsqueezeKernel::UnsqueezeKernel(Context* context)
        :Kernel(context){
            kernel_fname_ = "../opengl/nn/glsl/unsqueeze.glsl";
        }

    void UnsqueezeKernel::SetupAttr(const dlxnet::Attribute& attr){
        auto& unsqueeze_params = attr.unsqueeze_attr();
        for(auto item:unsqueeze_params.axes()){
            axes_.emplace_back(item);
        }

        // check axes
        // only support for pytorch due to torch.unsqueeze
        CHECK_EQ(axes_.size(), 1);
    }

    void UnsqueezeKernel::Compute(TensorList& inputs, TensorList& outputs){
        DLOG(INFO)<<"UnsqueezeKernel Inputs: "<<inputs.size();
        program_->Activate();
        auto input_image = inputs[0]->device<Texture>();

        SetFrameBuffer(outputs);
        SetVertexShader();
        // program_->set_vec2i("axis", axis_[0], axis_[1]);

        // input
        {
            program_->set_image2D("input_image", input_image->id(),  0);
            OPENGL_CHECK_ERROR;
        }

        OPENGL_CALL(glClear(GL_COLOR_BUFFER_BIT));
        OPENGL_CALL(glDrawArrays(GL_TRIANGLES, 0, 6));
        glFinish();
    }

    void UnsqueezeKernel::InferOutputShape(TensorShapeList& input_shapes,
            TensorShapeList& output_shapes){
        // set output dformat first, then we can according
        // to dformat to infer output shape
        output_tensor_dformats_.emplace_back(dlxnet::TensorProto::ANY4);

        output_shapes.clear();
        output_shapes.resize(1);
        CHECK_EQ(input_shapes.size(), 1);
        output_shapes[0] = input_shapes[0];
        // That appending to the end is not supported
        CHECK_NE(axes_[0], input_shapes.size());
        output_shapes[0].insert(output_shapes[0].begin()+axes_[0], 1);
    }

    UnsqueezeKernel::~UnsqueezeKernel(){}

    REGISTER_KERNEL_WITH_NAME(UnsqueezeKernel, "Unsqueeze");
}//namespace opengl
