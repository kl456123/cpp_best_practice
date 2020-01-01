#include <memory>
#include <string>
#include "core/backend.h"

#include "backends/cpu/cpu_backend.h"
#include "core/tensor.h"
#include "core/pool.h"



typedef Tensor::DataType DataType;
void* CPUPool::Malloc(size_t size, int alignment)
{
    return port::AlignMalloc(size, alignment);
}


CPUBackend::CPUBackend(Backend::ForwardType type):Backend(type){
    mPool.reset(new CPUPool());
}



void CPUBackend::Alloc(Tensor* tensor){
    tensor->set_host(mPool->Alloc(tensor->buffer_size()));
}

CPUBackend::~CPUBackend(){
}

void CPUBackend::Recycle(Tensor* tensor){
    mPool->Recycle(static_cast<void*>(tensor->host()));
}


void RegisterCPUBackend(){
    shared_ptr<Backend> ptr;
    ptr.reset(new CPUBackend(Backend::ForwardType::CPU));
    InsertBackend(Backend::ForwardType::CPU, ptr);
}


