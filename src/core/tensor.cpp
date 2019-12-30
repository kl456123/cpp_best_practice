#include "core/tensor.h"
#include "core/backend.h"
#include "core/port.h"
#include "core/define.h"



Tensor::Tensor(const std::vector<int>& tensor_shape, Tensor::DataType data_type, bool alloc){
    mSize = ComputeSize(tensor_shape);
    mShape = tensor_shape;
    mDevice = nullptr;
    mHost = nullptr;
    mOwnMemory = false;
    mDataType = data_type;

    if(alloc){
        mHost = port::AlignMalloc(mSize, MEMORY_ALIGN_DEFAULT);
        mOwnMemory = true;
    }
}

Tensor::~Tensor(){
    if(mOwnMemory){
        port::AlignFree(mHost);
    }
}


Tensor::Tensor(const std::vector<int>& tensor_shape, Tensor::DataType data_type, void* user_data){
    mSize = ComputeSize(tensor_shape);
    mDevice = nullptr;
    mShape = tensor_shape;
    mDataType = data_type;

    bool alloc = user_data==nullptr;
    if(alloc){
        mHost = port::AlignMalloc(mSize, MEMORY_ALIGN_DEFAULT);
        mOwnMemory = true;
    }else{

        mHost = user_data;
        mOwnMemory = false;
    }
}

Tensor::Tensor(const std::vector<int>& tensor_shape, Tensor::DataType data_type, Backend* backend){
    mSize = ComputeSize(tensor_shape);
    mShape = tensor_shape;
    mDevice = nullptr;
    mHost = nullptr;
    mDataType = data_type;

    if(backend!=nullptr){
        backend->Alloc(this);
        mOwnMemory = true;
    }
    else{
        mOwnMemory = false;
    }

}



// Tensor* Tensor::Create(const std::vector<int>& tensor_shape, Tensor::DataType data_type){
// // allocate to cpu default
// return new Tensor(tensor_shape, data_type, true);
// }


template<typename T>
Tensor* Tensor::Ones(const std::vector<int>& tensor_shape, Tensor::DataType data_type){
    Tensor* tensor = new Tensor(tensor_shape, data_type, true);
    Tensor::Filler::FillTensorByVal<T>(tensor, static_cast<T>(1.0));
}
template<typename T>
Tensor* Tensor::Zeros(const std::vector<int>& tensor_shape, Tensor::DataType data_type){
    auto tensor = new Tensor(tensor_shape, data_type, true);
    Tensor::Filler::FillTensorByVal<T>(tensor, static_cast<T>(0.0));
}
template<typename T>
Tensor* Tensor::Random(const std::vector<int>& tensor_shape,Tensor::DataType data_type){
    auto tensor = new Tensor(tensor_shape, data_type, true);
    Tensor::Filler::FillTensorRandomly<T>(tensor);
}
