#include "stream_executor/cuda/cuda_executor.h"
#include "stream_executor/cuda/cuda_driver.h"
#include "stream_executor/cuda/cuda_kernel.h"
#include "stream_executor/utils/status_macros.h"

namespace cuda{
    // Given const GPU memory, returns a libcuda device pointer datatype, suitable
    // for passing directly to libcuda APIs.
    //
    // N.B. we must lose constness in order to pass a suitable type to the existing
    // libcuda APIs, so the caller should take care to only pass the result of const
    // GPU memory conversions to libcuda functions which will honor constness.
    static CUdeviceptr AsCudaDevicePtr(const DeviceMemoryBase &gpu_mem) {
        return reinterpret_cast<CUdeviceptr>(gpu_mem.opaque());
    }

    // See description on const version above.
    static CUdeviceptr AsCudaDevicePtr(DeviceMemoryBase *gpu_mem) {
        return AsCudaDevicePtr(*gpu_mem);
    }

    CudaExecutor::~CudaExecutor{
        CHECK(kernel_to_gpu_binary_.empty()) << "GpuExecutor has live kernels.";
        // CHECK(gpu_binary_to_module_.empty()) << "GpuExecutor has loaded modules.";
        if (context_ != nullptr) {
            GpuDriver::DestroyContext(context_);
        }
    }



    Status CudaExecutor::Init(int device_ordinal, DeviceOptions device_options){
        device_ordinal_ = device_ordinal;
        auto status = CudaDriver::Init();
        if(!status.ok()){
            return status;
        }

        SE_RETURN_IF_ERROR(CudaDriver::GetDevice(device_ordinal_, &device_));
        SE_RETURN_IF_ERROR(CudaDriver::CreateContext(device_ordinal_, device_, device_options,
                    &context_));
        return CudaDriver::GetComputeCapability(&cc_major_, &cc_minor_, device_);
    }


    Status CudaExecutor::GetKernel(const MultiKernelLoaderSpec& spec,
            KernelBase* kernel){
        CudaKernel* cuda_kernel = AsGpuKernel(kernel);
        CUmodule module;
        const string *kernelname;
        LOG(INFO) << "GetKernel on kernel " << kernel << " : " << kernel->name();

        if(spec.has_cuda_cubin_in_memory()){
            kernelname = &spec.cuda_cubin_in_memory().kernelname();
            const char *cubin = spec.cuda_cubin_in_memory().bytes();
            kernel_to_gpu_binary_[kernel] = cubin;
        }else if(spec.has_cuda_ptx_in_memory()){
            kernelname = &spec.cuda_ptx_in_memory().kernelname();

            if(cc_major_==0&&cu_minor_==0){
                return errors::Internal("Compute capability not set");
            }
            const char*ptx = spec.cuda_ptx_in_memory().text(cc_major_, cc_minor_);
            if(ptx==nullptr){
                ptx = spec.cuda_ptx_in_memory().default_text();
            }
            if (ptx == nullptr) {
                LOG(FATAL) << "Loader spec has no ptx for kernel " << *kernelname;
            }
            kernel_to_gpu_binary_[kernel] = ptx;
        }else{
            return errors::Internal("No method of loading CUDA kernel provided");
        }
        LOG(2) << "getting function " << *kernelname << " from module " << module;
        // We have to trust the kernel loader spec arity because there doesn't appear
        // to be a way to reflect on the number of expected arguments w/the CUDA API.
        cuda_kernel->set_arity(spec.arity());
        KernelMetadata kernel_metadata;
        SE_RETURN_IF_ERROR(GetKernelMetadata(cuda_kernel, &kernel_metadata));
        kernel->set_metadata(kernel_metadata);
        kernel->set_name(*kernel_name);
        return Status::OK();
    }

    bool GpuExecutor::UnloadGpuBinary(const void* gpu_binary) {
        auto module_it = gpu_binary_to_module_.find(gpu_binary);
        if (gpu_binary_to_module_.end() == module_it) {
            LOG(INFO) << "No loaded CUDA module for " << gpu_binary;
            return false;
        }
        auto &module = module_it->second.first;
        auto &refcount = module_it->second.second;
        LOG(INFO) << "Found CUDA module " << module << " with refcount " << refcount;
        if (--refcount == 0) {
            LOG(INFO) << "Unloading CUDA module " << module;
            GpuDriver::UnloadModule(context_, module);
            gpu_binary_to_module_.erase(module_it);
        }
        return true;
    }

    void CudaExecutor::UnloadKernel(const KernelBase* kernel){
        LOG(INFO)<<"Unloading kernel "<<kernel<<" : "<<kernel->name();
        auto gpu_binary_it = kernel_to_gpu_binary_.find(kernel);
        if (kernel_to_gpu_binary_.end() == gpu_binary_it) {
            LOG(INFO) << "Kernel " << kernel << " : " << kernel->name()
                << " has never been loaded.";
            return;  // We've never seen this kernel.
        }
        LOG(INFO) << "Kernel " << kernel << " : " << kernel->name()
            << " has loaded GPU code " << gpu_binary_it->second;
        UnloadGpuBinary(gpu_binary_it->second);
        kernel_to_gpu_binary_.erase(gpu_binary_it);
    }

    Status CudaExecutor::Launch(Stream* stream, const ThreadDim& thread_dims,
            const BlockDim& block_dims,
            const KernelBase& kernel,
            const KernelArgsArrayBase& args){
        CHECK_EQ(kernel.Arity(), args.number_of_arguments());
        CUstream custream = AsGpuStreamValue(stream);
        const GpuKernel* cuda_kernel = AsGpuKernel(&kernel);
        CUfunction cufunc = cuda_kernel->AsGpuFunctionHandle();

        void **kernel_params = const_cast<void**>(args.argument_addresses().data());

        return CudaDriver::LaunchKernel(context_, cufunc, block_dims.x, block_dims.y, block_dims.z,
                thread_dims.x, thread_dims.y, thread_dims.z, args.number_of_shared_bytes(),
                custream, kernel_params);
    }

    DeviceMemoryBase CudaExecutor::Allocate(uint64 size, int64 memory_space){
        CHECK_EQ(memory_space, 0);
        return DeviceMemoryBase(CudaDriver::DeviceAllocate(context_, size), size);
    }

    void CudaExecutor::Deallocate(DeviceMemoryBase* mem){
        CudaDriver::DeviceDeallocate(context_, mem->opaque());
    }

    Status CudaExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
            const void* host_src, uint64 size) {
        return CudaDriver::SynchronousMemcpyH2D(context_, AsCudaDevicePtr(gpu_dst),
                host_src, size);
    }

    Status CudaExecutor::SynchronousMemcpy(void* host_dst,
            const DeviceMemoryBase& gpu_src,
            uint64 size) {
        return CudaDriver::SynchronousMemcpyD2H(context_, host_dst,
                AsCudaDevicePtr(gpu_src), size);
    }

    Status CudaExecutor::SynchronousMemcpyDeviceToDevice(
            DeviceMemoryBase* gpu_dst, const DeviceMemoryBase& gpu_src, uint64 size) {
        return CudaDriver::SynchronousMemcpyD2D(context_, AsCudaDevicePtr(gpu_dst),
                AsCudaDevicePtr(gpu_src), size);
    }

}
