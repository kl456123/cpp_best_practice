#include "opengl/nn/kernels/flatten.h"

#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"


namespace opengl{
    FlattenKernel::FlattenKernel(Context* context)
        :Kernel(context){
            kernel_fname_ = "../opengl/nn/glsl/flatten.glsl";
        }

    void FlattenKernel::SetupAttr(const dlxnet::Attribute& attr){
        auto& flatten_params = attr.flatten_attr();
        axis_ = flatten_params.axis();

        // single output
        output_tensor_dformats_.emplace_back(dlxnet::TensorProto::NHWC4);
    }

    void FlattenKernel::Compute(TensorList& inputs, TensorList& outputs){
        DLOG(INFO)<<"FlattenKernel Inputs: "<<inputs.size();
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

    void FlattenKernel::InferOutputShape(TensorShapeList& input_shapes,
            TensorShapeList& output_shapes){
        CHECK_EQ(input_shapes.size(), 1);
        output_shapes.clear();
        output_shapes.resize(1);
        CHECK_EQ(input_shapes[0].size(), 4);
        CHECK_EQ(output_shapes[0].size(), 0);

        int num_elements=1;
        for(int i=0;i<input_shapes[0].size();++i){
            num_elements*=input_shapes[0][i];
            output_shapes[0].emplace_back(1);
        }
        output_shapes[0][3]=num_elements;
    }

    FlattenKernel::~FlattenKernel(){}

    REGISTER_KERNEL_WITH_NAME(FlattenKernel, "Flatten");
}//namespace opengl
