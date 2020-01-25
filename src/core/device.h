#ifndef CORE_DEVICE_H_
#define CORE_DEVICE_H_
#include "core/platform/env.h"

class DeviceContext{
    public:
        ~DeviceContext()override{}
        virtual void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                Tensor* device_tensor)const{
            THROW_ERROR("Unrecognized device type in CPU-to-device Copy");
        }

        virtual void CopyTensorInSameDevice()const{
            THROW_ERROR("Copy in same device not implemented");
        }

        virtual void CopyDeviceTensorToCPU(const Tensor* device_tensor,
                std::string tensor_name, Device* device, Tensor* cpu_tensor){
            THROW_ERROR("Unrecognized device type in device to CPU Copy");
        }
};

class DeviceBase{
    public:
        explicit DeviceBase(Env* env):env_(env){}
        virtual ~DeviceBase();

        Env* env()const {return env_;}

    private:
        Env* const env_;
};




#endif
