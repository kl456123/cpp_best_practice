#include <memory>
#include <string>
#include "core/backend.h"

#include "backends/cpu/cpu_backend.h"


class Backend;
class Tensor;

typedef Tensor::DataType DataType;


CPUBackend::CPUBackend(){
    mPool.reset(new Pool());
}



void CPUBackend::Alloc(Tensor* tensor, DataType data_type){
    switch(data_type){
        case DataType::FLOAT32:
            tensor->mHost = mPool->Alloc<float>(tensor->mSize);
            break;
        case DataType::DOUBLE:
            tensor->mHost = mPool->Alloc<double>(tensor->mSize);
            break;
        case DataType::INT32:
            tensor->mHost = mPool->Alloc<int32_t>(tensor->mSize);
            break;
        case DataType::INT8:
            tensor->mHost = mPool->Alloc<int8_t>(tensor->mSize);
            break;
        default:
            ;
    }
}

CPUBackend::~CPUBackend(){
}

void CPUBackend::Clear(){
    mPool->Clear();
}

void CPUBackend::Recycle(Tensor* tensor, DataType data_type){
    switch(data_type){
        case DataType::FLOAT32:
            mPool->Recycle<float>(tensor->mHost);
            break;
        case DataType::DOUBLE:
            mPool->Recycle<double>(tensor->mHost);
            break;
        case DataType::INT32:
            mPool->Recycle<int32_t>(tensor->mHost);
            break;
        case DataType::INT8:
            mPool->Recycle<int8_t>(tensor->mHost);
            break;
        default:
            ;
    }
}


void RegisterCPUBackend(){
    shared_ptr<Backend> ptr;
    ptr.reset(new CPU());
    InsertBackend(Backend::ForwardType::CPU, ptr);
}
