#include <cstring>

#include "context.h"
#include "macros.h"


void Context::Compute(std::initializer_list<size_t> dim_sizes){
    auto ptr = dim_sizes.begin();
    glDispatchCompute(ptr[0], ptr[1], ptr[2]);
}

Context::Context(Allocator* allocator)
    :allocator_(allocator){
    }

void Context::CopyCPUTensorToDevice(const Tensor* cpu_tensor, Tensor* device_tensor){
    // do some necessary check first

    // check cpu_tensor is in cpu
    CHECK(cpu_tensor->is_host());

    // check same size
    CHECK(cpu_tensor->size()==device_tensor->size());
    CHECK(cpu_tensor->num_elements()==device_tensor->num_elements());

    Buffer* device_buffer = reinterpret_cast<Buffer*>(device_tensor->device());
    ::memcpy(device_buffer->Map(GL_MAP_WRITE_BIT),cpu_tensor->host(),
            cpu_tensor->size());

    device_buffer->UnMap();
}


void Context::CopyDeviceTensorToCPU(const Tensor* device_tensor, Tensor* cpu_tensor){
    // do some necessary check first

    // check cpu_tensor is in cpu
    CHECK(cpu_tensor->is_host());

    // check same size
    CHECK(cpu_tensor->num_elements()==device_tensor->num_elements());

    Buffer* device_buffer = reinterpret_cast<Buffer*>(device_tensor->device());
    ::memcpy(cpu_tensor->host(), device_buffer->Map(GL_MAP_READ_BIT),
            cpu_tensor->size());

    device_buffer->UnMap();
}
