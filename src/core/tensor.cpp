#include "core/tensor.h"
#include "core/backend.h"
#include "core/port.h"
#include "core/macros.h"

// used for dump
#include <iostream>
#include <fstream>
#include <assert.h>

void Tensor::Init(const std::vector<int>& tensor_shape_tmp, DataType data_type){
    // default init tensor here to reduce constructor burden
    std::vector<int> tensor_shape(tensor_shape_tmp);
    int remain_axis = 4-tensor_shape.size();
    for(int i=0;i<remain_axis;i++){
        tensor_shape.insert(tensor_shape.begin(), 1);
    }
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

void Tensor::CopyFromTensor(Tensor* other){
    if(mHost==nullptr){
        Backend* backend = ExtractBackend(Backend::ForwardType::CPU);
        backend->Alloc(this);
    }
    if(other->host()==nullptr){
        other->CopyToHost();
    }

    assert(mBufferSize==other->buffer_size());

    // copy from other host to mHost
    memcpy(mHost, other->host(), mBufferSize);
}

Tensor::~Tensor(){
    if(mOwnMemory){
        assert(mHost!=nullptr);
        port::AlignFree(mHost);
    }

    if(mDevice!=nullptr){
        Backend* backend = ExtractBackend(mDeviceType);
        backend->Recycle(this);
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
    }
}

Tensor::Tensor(const std::vector<int>& tensor_shape, Tensor::DataType data_type, Backend* backend){
    Init(tensor_shape, data_type);
    mDeviceType = backend->type();

    if(backend!=nullptr){
        backend->Alloc(this);
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

template<typename T>
void Tensor::Dump(const std::string& file_name){
    ofstream file;
    file.open(file_name, std::ofstream::out);
    if(file.fail()){
        std::cout<<"Open file Error: "<<file_name<<std::endl;
        return;
    }
    int batch_size = mShape[0];
    int output_channels = mShape[1];

    for(int i=0;i<mShape[0];i++){
        for(int j=0;j<mShape[1];j++){
            for(int k=0;k<mShape[2];k++){
                for(int l=0;l<mShape[3];l++){
                    int index = Offset(i,j,k,l);
                    file<<((T*)mHost)[index] <<" ";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }

    file.close();

}
template void Tensor::Dump<float>(const std::string& filename);
