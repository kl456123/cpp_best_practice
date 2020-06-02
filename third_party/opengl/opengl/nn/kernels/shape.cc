#include "opengl/nn/kernels/shape.h"

#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"


namespace opengl{
    ShapeKernel::ShapeKernel(Context* context)
        :Kernel(context){
            // kernel_fname_ = "../opengl/nn/glsl/shape.glsl";
        }

    void ShapeKernel::SetupAttr(const dlxnet::Attribute& attr){
    }

    void ShapeKernel::Compute(TensorList& inputs, TensorList& outputs){
        DLOG(INFO)<<"ShapeKernel Inputs: "<<inputs.size();
    }

    void ShapeKernel::InferOutputShape(TensorShapeList& input_shapes,
            TensorShapeList& output_shapes){
        // set output dformat first, then we can according
        // to dformat to infer output shape
        output_tensor_dformats_.emplace_back(dlxnet::TensorProto::NHWC4);

        output_shapes.clear();
        output_shapes.resize(1);
        CHECK_EQ(input_shapes.size(), 1);
        const int dims_size = input_shapes[0].size();
        output_shapes[0] = {dims_size};
    }

    ShapeKernel::~ShapeKernel(){}

    REGISTER_KERNEL_WITH_NAME(ShapeKernel, "Shape");
}//namespace opengl
