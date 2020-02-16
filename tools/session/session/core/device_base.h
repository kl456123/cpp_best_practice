#ifndef SESSION_CORE_DEVICE_BASE_H_
#define SESSION_CORE_DEVICE_BASE_H_
#include "session/utils/env.h"
#include "session/utils/logging.h"
#include "session/core/tensor.h"
#include "session/utils/errors.h"
#include "tensor.pb.h"
#include "session/core/allocator.h"

class DeviceAttributes;
class DeviceContext;
class Allocator;
class AllocatorAttributes;
class Device;

class DeviceContext{
    public:
        virtual ~DeviceContext(){}
        virtual void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                Tensor* device_tensor)const{
            errors::Internal("Unrecognized device type in CPU-to-device Copy");
        }

        virtual void CopyTensorInSameDevice(const Tensor* input_tensor,
                Device* device, Tensor* output_tensor)const{
            errors::Unimplemented("Copy in same device not implemented.");
        }

        virtual void CopyDeviceTensorToCPU(const Tensor* device_tensor,
                std::string tensor_name, Device* device, Tensor* cpu_tensor){
            errors::Internal("Unrecognized device type in device-to-CPU Copy");
        }
};

class DeviceBase{
    public:
        explicit DeviceBase(Env* env):env_(env){}
        virtual ~DeviceBase();

        Env* env()const {return env_;}
        virtual Allocator* GetAllocator(AllocatorAttributes attr){
            LOG(FATAL) << "GetAllocator() is not implemented.";
            return nullptr;
        }
        // Unimplemented by default
        virtual const DeviceAttributes& attributes() const;
        virtual const std::string& name() const;

        // Materializes the given TensorProto into 'tensor' stored in Device
        // memory.  Most devices will want to override this.
        //
        // TODO(vrv): We should be able to put this function into
        // OpKernelContext and handle the copies from device memory via send
        // and receive nodes, instead of requiring that each device handle
        // the copies here as well as in copy ops.
        virtual Status MakeTensorFromProto(const TensorProto& tensor_proto,
                const AllocatorAttributes alloc_attrs,
                Tensor* tensor) {
            return errors::Internal("Device does not implement MakeTensorFromProto()");
        }

        // Copies `input_tensor` to `output_tensor`, where both tensors are on this
        // device. This function assumes that `output_tensor` has already been
        // allocated with a buffer that is large enough to hold `input_tensor`'s data.
        // Calls `done` from a device-specific thread after copy is finished, which
        // may be the same as calling thread.
        //
        // NOTE(ayushd): This function is for TensorFlow internal use only.  Deep copy
        // is discouraged and should not be used in OpKernels.
        virtual void CopyTensorInSameDevice(const Tensor* input_tensor,
                Tensor* output_tensor,
                const DeviceContext* device_context
                ) {
            errors::Internal("Device ", name(), " does not implement ",
                    "CopyTensorInSameDevice");
        }

    private:
        Env* const env_;
};




#endif
