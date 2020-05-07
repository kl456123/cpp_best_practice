#include "opengl/nn/kernels/const.h"
#include "opengl/core/kernel_registry.h"
#include "opengl/core/context.h"
#include "opengl/core/kernel.h"

namespace opengl{
    void ConstKernel::SetupAttr(const dlxnet::Attribute& attr){
        // get value and store it in tensor_
        auto& const_params = attr.const_attr();
        const_params.value();
    }

    ConstKernel::ConstKernel(Context* context)
        :Kernel(nullptr){
        }

    void ConstKernel::Compute(TensorList& inputs, TensorList& outputs){
        // set value from tensor_
    }

    void ConstKernel::InferOutputShape(TensorShapeList& inputs,
            TensorShapeList& outputs){
        // check input tensors is empty
        CHECK(inputs.size()==0)<<"Input should be empty in ConstKernel";

        // get shape from outputs
        // outputs.emplace_back();
    }

    ConstKernel::~ConstKernel(){
    }

    REGISTER_KERNEL_WITH_NAME(ConstKernel, "Const");
}//namespace opengl
