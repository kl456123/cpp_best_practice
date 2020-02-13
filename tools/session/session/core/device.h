#ifndef SESSION_CORE_DEVICE_H_
#define SESSION_CORE_DEVICE_H_
#include "session/core/device_base.h"
#include "session/core/op_kernel.h"
#include "session/core/types.h"
#include "session/utils/macros.h"
#include "device_attributes.pb.h"
#include "session/core/op_segment.h"

typedef int64_t Bytes;

class Device:public DeviceBase{
    public:
        Device(Env* env, const DeviceAttributes& device_attributes);
        ~Device()override;
        // Full name of this device (see top comment).
        const string& name() const override { return device_attributes_.name(); }
        const string& device_type() const { return device_attributes_.device_type(); }
        const DeviceAttributes& attributes() const override {
            return device_attributes_;
        }
        // Performs the actual compute function.
        //
        // Subclasses may override this function if they wish to perform
        // some initialization before each compute.
        virtual void Compute(OpKernel* op_kernel, OpKernelContext* context) {
            op_kernel->Compute(context);
        }
        // Blocks until all operations queued on the device at the time of
        // the call have completed.  Returns any error pending on the device
        // at completion.
        virtual Status Sync() = 0;
        // Returns the op segment of this device.  The caller can reuse op
        // kernels registered for the same session running on this device.
        OpSegment* op_segment() { return &op_seg_; }

        // Summarizes the status of this Device, for debugging.
        string DebugString() const { return device_attributes_.DebugString(); }

        virtual bool IsLocal() const { return true; }
        // Assembles the parameter components into a complete DeviceAttributes value.
        static DeviceAttributes BuildDeviceAttributes(
                const string& name, DeviceType device, Bytes memory_limit,
                const DeviceLocality& locality, const string& physical_device_desc);

        static DeviceAttributes BuildDeviceAttributes(
                const string& name, DeviceType device, Bytes memory_limit,
                const DeviceLocality& locality) {
            // Pass in an empty string as physical device name.
            return BuildDeviceAttributes(name, device, memory_limit, locality, "");
        }

    private:
        const DeviceAttributes device_attributes_;

        // op_seg_ maps session handle and op name to OpKernel objects.
        OpSegment op_seg_;

        DISALLOW_COPY_AND_ASSIGN(Device);
};

#endif
