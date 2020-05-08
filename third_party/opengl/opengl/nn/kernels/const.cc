#include "opengl/nn/kernels/const.h"
#include "opengl/core/kernel_registry.h"
#include "opengl/core/context.h"
#include "opengl/core/kernel.h"

namespace opengl{
    void ConstKernel::SetupAttr(const dlxnet::Attribute& attr){
        // get value and store it in tensor_
        auto& const_tensor = attr.const_attr().value();
        // get tensor shape
        int num_elements=1;
        for(auto& dim:const_tensor.dims()){
            tensor_shape_.emplace_back(dim);
            num_elements*=dim;
        }
        num_elements_ = num_elements;
        // get tensor data
        switch(const_tensor.data_type()){
            case dlxnet::TensorProto::FLOAT32:
                {
                    tensor_data_ = new float[num_elements_];
                    float* target_data = static_cast<float*>(tensor_data_);
                    for(int i=0;i<num_elements;++i){
                        target_data[i] = const_tensor.float_data(i);
                    }
                    break;
                }

            case dlxnet::TensorProto::INT32:
                {
                    tensor_data_ = new int[num_elements_];
                    int* target_data = static_cast<int*>(tensor_data_);
                    for(int i=0;i<num_elements;++i){
                        target_data[i] = const_tensor.int32_data(i);
                    }
                    break;
                }

            default:
                LOG(FATAL)<<"unsupported const type: "<<const_tensor.data_type();
        }
    }

    ConstKernel::ConstKernel(Context* context)
        :Kernel(nullptr){
        }

    void ConstKernel::Compute(TensorList& inputs, TensorList& outputs){
        // set value from tensor_
        // for now just copy host memory data to device
        CHECK(inputs.size()==0)<<"Input should be empty in ConstKernel";
        outputs[0];
    }

    void ConstKernel::InferOutputShape(TensorShapeList& inputs,
            TensorShapeList& outputs){
        // check input tensors is empty
        CHECK(inputs.size()==0)<<"Input should be empty in ConstKernel";

        // get shape from outputs
        outputs.emplace_back(tensor_shape_);
    }

    ConstKernel::~ConstKernel(){
    }

    REGISTER_KERNEL_WITH_NAME(ConstKernel, "Const");
}//namespace opengl
