#include "core/tensor.h"
#include "core/backend.h"
#include "core/port.h"
#include "core/define.h"

void Tensor::Init(const std::vector<int>& tensor_shape, DataType data_type){
    // default init tensor here to reduce constructor burden
    mSize = ComputeSize(tensor_shape);
    ComputeStride(tensor_shape, mStride);
    mBufferSize = ComputeBufferSize(mSize, data_type);
    mShape = tensor_shape;
    mDevice = nullptr;
    mHost = nullptr;
    mOwnMemory = false;
    mDataType = data_type;
    mDeviceType = Backend::ForwardType::CPU;
}



Tensor::Tensor(const std::vector<int>& tensor_shape, Tensor::DataType data_type, bool alloc){
    Init(tensor_shape, data_type);

    if(alloc){
        mHost = port::AlignMalloc(buffer_size(), MEMORY_ALIGN_DEFAULT);
        mOwnMemory = true;
    }
}

Tensor::~Tensor(){
    if(mOwnMemory){
        port::AlignFree(mHost);
        if(mDevice!=nullptr){
            // std::cout<<"free device"<<std::endl;
            Backend* backend = ExtractBackend(mDeviceType);
            backend->Recycle(this);
        }
    }
}


Tensor::Tensor(const std::vector<int>& tensor_shape, Tensor::DataType data_type, void* user_data){
    Init(tensor_shape, data_type);

    bool alloc = user_data==nullptr;
    if(alloc){
        mHost = port::AlignMalloc(buffer_size(), MEMORY_ALIGN_DEFAULT);
        mOwnMemory = true;
    }else{

        mHost = user_data;
        mOwnMemory = false;
    }
}

Tensor::Tensor(const std::vector<int>& tensor_shape, Tensor::DataType data_type, Backend::ForwardType type){
    Init(tensor_shape, data_type);

    auto backend = ExtractBackend(type);
    mDeviceType = type;
    if(backend!=nullptr){
        backend->Alloc(this);
        mOwnMemory = true;
    }
    else{
        mOwnMemory = false;
    }
}

Tensor::Tensor(const std::vector<int>& tensor_shape, Tensor::DataType data_type, Backend* backend){
    Init(tensor_shape, data_type);
    mDeviceType = backend->type();

    if(backend!=nullptr){
        backend->Alloc(this);
        mOwnMemory = true;
    }
    else{
        mOwnMemory = false;
    }

}




Tensor* Tensor::Ones(const std::vector<int>& tensor_shape, Tensor::DataType data_type){
    Tensor* tensor = new Tensor(tensor_shape, data_type, true);
    Tensor::Filler::FillTensorByVal(tensor, 1.0);
    return tensor;
}
Tensor* Tensor::Zeros(const std::vector<int>& tensor_shape, Tensor::DataType data_type){
    auto tensor = new Tensor(tensor_shape, data_type, true);
    Tensor::Filler::FillTensorByVal(tensor, 0.0);
    return tensor;
}

Tensor* Tensor::Random(const std::vector<int>& tensor_shape,Tensor::DataType data_type){
    auto tensor = new Tensor(tensor_shape, data_type, true);
    Tensor::Filler::FillTensorRandomly(tensor);
    return tensor;
}


template<typename T>
void Tensor::Print(int size){
    if(size==-1){
        size=mSize;
    }
    int batch_size = mShape[0];
    int output_channels = mShape[1];

    for(int i=0;i<mShape[0];i++){
        for(int j=0;j<mShape[1];j++){
            for(int k=0;k<mShape[2];k++){
                for(int l=0;l<mShape[3];l++){
                    int index = Offset(i,j,k,l);
                    if(index>=size){
                        return;
                    }
                    std::cout<<((T*)mHost)[index] <<" ";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }
}
template void Tensor::Print<float>(int);
