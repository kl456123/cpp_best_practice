#include "opengl/nn/kernels/flatten.h"

#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"
#include "opengl/utils/util.h"


namespace opengl{
    FlattenKernel::FlattenKernel(Context* context)
        :Kernel(context){
            // kernel_fname_ = "../opengl/nn/glsl/flatten.glsl";
            kernel_fname_ = "../opengl/nn/glsl/reshape.glsl";
        }

    void FlattenKernel::SetupAttr(const dlxnet::Attribute& attr){
        auto& flatten_params = attr.flatten_attr();
        axis_ = flatten_params.axis();


    }

    void FlattenKernel::Compute(TensorList& inputs, TensorList& outputs){
        DLOG(INFO)<<"FlattenKernel Inputs: "<<inputs.size();
        program_->Activate();
        auto input_image = inputs[0]->device<Texture>();

        program_->SetRetVal(outputs);
        program_->set_vec4i("input_shape", AmendShape(inputs[0]->shape()));
        program_->set_vec4i("output_shape", AmendShape(outputs[0]->shape()));

        // input
        {
            program_->set_image2D("input_image", input_image->id(),  0);
            OPENGL_CHECK_ERROR;
        }

        program_->Run();
    }

    void FlattenKernel::InferOutputShape(TensorShapeList& input_shapes,
            TensorShapeList& output_shapes){
        // set output dformat first, then we can according
        // to dformat to infer output shape
        output_tensor_dformats_.emplace_back(dlxnet::TensorProto::ANY4);

        // single output
        CHECK_EQ(input_shapes.size(), 1);
        output_shapes.clear();
        output_shapes.resize(1);
        CHECK_EQ(input_shapes[0].size(), 4);
        CHECK_EQ(output_shapes[0].size(), 0);
        const int start_dim = axis_;

        int num_elements=1;
        for(int i=start_dim;i<input_shapes[0].size();++i){
            num_elements*=input_shapes[0][i];
        }
        for(int i=0;i<start_dim;++i){
            output_shapes[0].emplace_back(input_shapes[0][i]);
        }
        output_shapes[0].emplace_back(num_elements);
    }

    FlattenKernel::~FlattenKernel(){}

    REGISTER_KERNEL_WITH_NAME(FlattenKernel, "Flatten");
}//namespace opengl
