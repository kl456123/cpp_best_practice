#ifndef SESSION_CORE_DEVICE_BASE_H_
#define SESSION_CORE_DEVICE_BASE_H_
#include "session/utils/env.h"
#include "session/utils/logging.h"
#include "session/core/tensor.h"
#include "session/utils/errors.h"

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
        virtual Allocator* GetAllocator(){
            LOG(FATAL) << "GetAllocator() is not implemented.";
            return nullptr;
        }
        // Unimplemented by default
        virtual const DeviceAttributes& attributes() const;
        virtual const std::string& name() const;

    private:
        Env* const env_;
};




#endif
