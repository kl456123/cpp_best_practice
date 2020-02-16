#ifndef SESSION_CORE_THREADPOOL_DEVICE_H_
#define SESSION_CORE_THREADPOOL_DEVICE_H_
#include "session/core/device.h"
#include "session/core/session_options.h"
#include "session/core/tensor.h"
#include "tensor.pb.h"

class ThreadPoolDevice: public Device{
	public:
  ThreadPoolDevice(const SessionOptions& options, const string& name,
                   Bytes memory_limit, const DeviceLocality& locality,
                   Allocator* allocator);
  ~ThreadPoolDevice() override;

  Allocator* GetAllocator(AllocatorAttributes attr) override;
  // Allocator* GetScopedAllocator(AllocatorAttributes attr,
                                // int64 step_id) override;
  // ScopedAllocatorMgr* GetScopedAllocatorMgr() const override {
    // return scoped_allocator_mgr_.get();
  // }
  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override;
  void CopyTensorInSameDevice(const Tensor* input_tensor, Tensor* output_tensor,
                              const DeviceContext* device_context) override;

  Status Sync() override { return Status::OK(); }

 private:
  Allocator* allocator_;  // Not owned
  // std::unique_ptr<ScopedAllocatorMgr> scoped_allocator_mgr_;
};



#endif
