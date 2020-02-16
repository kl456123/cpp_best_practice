#include "session/core/threadpool_device.h"

ThreadPoolDevice::ThreadPoolDevice(const SessionOptions& options, const string& name,
        Bytes memory_limit, const DeviceLocality& locality,
        Allocator* allocator):Device(options.env, Device::BuildDeviceAttributes(
                name, DEVICE_CPU, memory_limit, locality)), allocator_(allocator){
}


ThreadPoolDevice::~ThreadPoolDevice() {}

Allocator* ThreadPoolDevice::GetAllocator(AllocatorAttributes attr) {
    return allocator_;
}


Status ThreadPoolDevice::MakeTensorFromProto(
        const TensorProto& tensor_proto, const AllocatorAttributes alloc_attrs,
        Tensor* tensor) {
    if (tensor_proto.dtype() > 0 && tensor_proto.dtype() <= DataType_MAX) {
        // Tensor parsed(tensor_proto.dtype());
        // if (parsed.FromProto(allocator_, tensor_proto)) {
        // *tensor = std::move(parsed);
        return Status::OK();
        // }
    }
    return errors::InvalidArgument("Cannot parse tensor from proto: ",
            tensor_proto.DebugString());
}


void ThreadPoolDevice::CopyTensorInSameDevice(
        const Tensor* input_tensor, Tensor* output_tensor,
        const DeviceContext* device_context) {
    // if (input_tensor->NumElements() != output_tensor->NumElements()) {
    // done(errors::Internal(
    // "CPU->CPU copy shape mismatch: input=", input_tensor->shape(),
    // ", output=", output_tensor->shape()));
    // return;
    // }
    // tensor::DeepCopy(*input_tensor, output_tensor);
    Status::OK();
}
