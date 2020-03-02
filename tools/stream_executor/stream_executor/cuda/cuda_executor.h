#ifndef STREAM_EXECUTOR_CUDA_CUDA_EXECUTOR_H_
#define STREAM_EXECUTOR_CUDA_CUDA_EXECUTOR_H_
#include <memory>
#include <unordered_map>

#include "stream_executor/core/stream_executor_internal.h"

class CudaExecutor: public internal::StreamExecutorInterface{
    public:
        explicit GpuExecutor()
            : device_(0),
            context_(nullptr),
            device_ordinal_(0),
            cc_major_(0),
            cc_minor_(0),
            version_(0){}

        // See the corresponding StreamExecutor methods for method comments on the
        // following overrides.

        ~GpuExecutor() override;

        Status Init(int device_ordinal, DeviceOptions device_options) override;

        Status GetKernel(const MultiKernelLoaderSpec& spec,
                KernelBase* kernel) override;
        // (supported on CUDA only)
        void UnloadKernel(const KernelBase* kernel) override;
        Status Launch(Stream* stream, const ThreadDim& thread_dims,
                const BlockDim& block_dims, const KernelBase& k,
                const KernelArgsArrayBase& args) override;
        DeviceMemoryBase Allocate(uint64 size, int64 memory_space) override;
        void Deallocate(DeviceMemoryBase* mem) override;
        void* UnifiedMemoryAllocate(uint64 size) override {
            return GpuDriver::UnifiedMemoryAllocate(context_, size);
        }
        void UnifiedMemoryDeallocate(void* location) override {
            return GpuDriver::UnifiedMemoryDeallocate(context_, location);
        }
        Status SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                const void* host_src, uint64 size) override;

        Status SynchronousMemcpy(void* host_dst,
                const DeviceMemoryBase& gpu_src,
                uint64 size) override;
        Status SynchronousMemcpyDeviceToDevice(DeviceMemoryBase* gpu_dst,
                const DeviceMemoryBase& gpu_src,
                uint64 size) override;

        bool Memcpy(Stream* stream, void* host_dst, const DeviceMemoryBase& gpu_src,
                uint64 size) override;

        bool Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst, const void* host_src,
                uint64 size) override;

        bool MemcpyDeviceToDevice(Stream* stream, DeviceMemoryBase* gpu_dst,
                const DeviceMemoryBase& gpu_src,
                uint64 size) override;
        bool AllocateStream(Stream* stream) override;

        void DeallocateStream(Stream* stream) override;
        bool AllocateTimer(Timer* timer) override;

        void DeallocateTimer(Timer* timer) override;

        bool StartTimer(Stream* stream, Timer* timer) override;

        bool StopTimer(Stream* stream, Timer* timer) override;
        Status BlockHostUntilDone(Stream* stream) override;
        int PlatformDeviceCount() override { return GpuDriver::GetDeviceCount(); }
        bool DeviceMemoryUsage(int64* free, int64* total) const override;

        StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
            const override {
                return CreateDeviceDescription(device_ordinal_);
            }
        static Status
            CreateDeviceDescription(int device_ordinal, std::unique_ptr<DeviceDescription>*);
        std::unique_ptr<internal::KernelInterface> CreateKernelImplementation()
            override;

        std::unique_ptr<internal::StreamInterface> GetStreamImplementation() override;

        std::unique_ptr<internal::TimerInterface> GetTimerImplementation() override;
    private:
        // Collects metadata for the specified kernel.
        Status GetKernelMetadata(GpuKernel* cuda_kernel,
                KernelMetadata* kernel_metadata);
        // Keeps track of the set of launched kernels. Currently used to suppress the
        // occupancy check on subsequent launches.
        std::set<CudaFunctionHandle> launched_kernels_;

        // Kernel -> loaded GPU binary. Many kernels may load the same binary.
        std::unordered_map<const KernelBase*, const void*> kernel_to_gpu_binary_;

        // Handle for session with the library/driver. Immutable post-initialization.
        CudaContext* context_;

        // The device ordinal value that this executor was initialized with; recorded
        // for use in getting device metadata. Immutable post-initialization.
        int device_ordinal_;

        // The major verion of the compute capability for device_.
        int cc_major_;

        // The minor verion of the compute capability for device_.
        int cc_minor_;

        // GPU ISA version for device_.
        int version_;

        SE_DISALLOW_COPY_AND_ASSIGN(CudaExecutor);
};


#endif
