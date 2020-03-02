#ifndef STREAM_EXECUTOR_CUDA_CUDA_KERNEL_H_
#define STREAM_EXECUTOR_CUDA_CUDA_KERNEL_H_
#include "stream_executor/core/stream_executor_internal.h"

class CudaKernel: public internal::KernelInterface{
    public:
        CudaKernel()
            : cuda_function_(nullptr),
            arity_(0),
            preferred_cache_config_(KernelCacheConfig::kNoPreference) {}

        // Note that the function is unloaded when the module is unloaded, and the
        // module that the function is contained in is owned by the GpuExecutor.
        ~CudaKernel() override {}
        // As arity cannot be reflected upon using the CUDA API, the arity is
        // explicitly set during the GpuExecutor::GetKernel initialization process.
        void set_arity(unsigned arity) { arity_ = arity; }
        unsigned Arity() const override { return arity_; }

        // Returns the GpuFunctionHandle value for passing to the CUDA API.
        CudaFunctionHandle AsCudaFunctionHandle() const {
            DCHECK(cuda_function_ != nullptr);
            return const_cast<CudaFunctionHandle>(cuda_function_);
        }

        // Returns the slot that the GpuFunctionHandle is stored within for this
        // object, for the CUDA API which wants to load into a GpuFunctionHandle*.
        CudaFunctionHandle* cuda_function_ptr() { return &cuda_function_; }
    private:
        CudaFunctionHandle cuda_function_;  // Wrapped CUDA kernel handle.
        unsigned arity_;  // Number of formal parameters the kernel takes.
};

// Given a platform-independent kernel datatype, returns the (const) internal
// CUDA platform implementation pointer.
inline const CudaKernel* AsGpuKernel(const KernelBase* kernel) {
  return static_cast<const CudaKernel*>(kernel->implementation());
}

// Given a platform-independent kernel datatype, returns the (non-const)
// internal CUDA platform implementation pointer.
inline CudaKernel* AsGpuKernel(KernelBase* kernel) {
  return static_cast<CudaKernel*>(kernel->implementation());
}


#endif
