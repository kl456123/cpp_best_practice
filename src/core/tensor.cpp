#include "core/tensor.h"
#include "core/backend.h"



Tensor::Tensor(const std::vector<int>& tensor_shape, Tensor::DataType data_type, bool alloc){
    mSize = ComputeSize(tensor_shape);
    mShape = tensor_shape;
    mDevice = nullptr;
    mHost = nullptr;
    mOwnMemory = false;
    mDataType = data_type;

    if(alloc){
        Backend* backend = ExtractBackend();
        mHost = backend->Alloc(mSize, mDataType);
        mOwnMemory = true;
    }
}

Tensor::~Tensor(){
    if(mOwnMemory){
        auto backend = ExtractBackend(Backend::ForwardType::CPU);
        backend->Recycle(this, mDataType);
    }
}





Tensor* Create(const std::vector<int>& tensor_shape, Tensor::DataType data_type){
    // allocate to cpu default
    return new Tensor(tensor_shape, data_type, true);
}


Tensor* Ones(const std::vector<int>& tensor_shape, Tensor::DataType data_type){
    Tensor* tensor = Tensor::Create(tensor_shape, data_type);
    Tensor::Filler::FillTensorByVal<float>(tensor, 1.0);
}

Tensor* Zeros(const std::vector<int>& tensor_shape, ){
    auto tensor = Tensor::Create(tensor_shape);
    Tensor::Filler::FillTensorByVal<>(tensor, 0.0);
}

Tensor* Random(const std::vector<int>& tensor_shape){
    auto tensor = Tensor::Create(tensor_shape);
    Tensor::Filler::FillTensorRandomly(tensor);
}
