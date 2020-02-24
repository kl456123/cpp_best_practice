#ifndef STREAM_EXECUTOR_CORE_INTERNAL_H_
#define STREAM_EXECUTOR_CORE_INTERNAL_H_
#include "stream_executor/utils/status.h"
#include "stream_executor/utils/errors.h"
#include "stream_executor/utils/macros.h"

#include "stream_executor/core/allocator_stats.h"
#include "stream_executor/core/device_description.h"
#include "stream_executor/core/device_memory.h"
#include "stream_executor/core/kernel.h"
#include "stream_executor/core/device_options.h"
#include "stream_executor/core/kernel_cache_config.h"
#include "stream_executor/core/kernel_spec.h"
#include "stream_executor/core/launch_dim.h"

#include <cstdint>

typedef uint64_t uint64;
class Stream;
class Timer;

namespace internal{
    // Platform-dependent interface class for the generic Events interface, in
    // the PIMPL style.
    class EventInterface {
        public:
            EventInterface() {}
            virtual ~EventInterface() {}

        private:
            DISALLOW_COPY_AND_ASSIGN(EventInterface);
    };

    // Pointer-to-implementation object type (i.e. the KernelBase class delegates to
    // this interface) with virtual destruction. This class exists for the
    // platform-dependent code to hang any kernel data/resource info/functionality
    // off of.
    class KernelInterface {
        public:
            // Default constructor for the abstract interface.
            KernelInterface() {}

            // Default destructor for the abstract interface.
            virtual ~KernelInterface() {}

            // Returns the number of formal parameters that this kernel accepts.
            virtual unsigned Arity() const = 0;

            // Sets the preferred cache configuration.
            virtual void SetPreferredCacheConfig(KernelCacheConfig config) = 0;

            // Gets the preferred cache configuration.
            virtual KernelCacheConfig GetPreferredCacheConfig() const = 0;

        private:
            DISALLOW_COPY_AND_ASSIGN(KernelInterface);
    };

    // Pointer-to-implementation object type (i.e. the Stream class delegates to
    // this interface) with virtual destruction. This class exists for the
    // platform-dependent code to hang any kernel data/resource info/functionality
    // off of.
    class StreamInterface {
        public:
            // Default constructor for the abstract interface.
            StreamInterface() {}

            // Default destructor for the abstract interface.
            virtual ~StreamInterface() {}

            // Returns the GPU stream associated with this platform's stream
            // implementation.
            //
            // WARNING: checks that the underlying platform is, in fact, CUDA or ROCm,
            // causing a fatal error if it is not. This hack is made available solely for
            // use from distbelief code, which temporarily has strong ties to CUDA or
            // ROCm as a platform.
            virtual void *GpuStreamHack() { return nullptr; }

            // See the above comment on GpuStreamHack -- this further breaks abstraction
            // for Eigen within distbelief, which has strong ties to CUDA or ROCm as a
            // platform, and a historical attachment to a programming model which takes a
            // stream-slot rather than a stream-value.
            virtual void **GpuStreamMemberHack() { return nullptr; }

        private:
            DISALLOW_COPY_AND_ASSIGN(StreamInterface);
    };

    // Pointer-to-implementation object type (i.e. the Timer class delegates to
    // this interface) with virtual destruction. This class exists for the
    // platform-dependent code to hang any timer data/resource info/functionality
    // off of.
    class TimerInterface {
        public:
            // Default constructor for the abstract interface.
            TimerInterface() {}

            // Default destructor for the abstract interface.
            virtual ~TimerInterface() {}

            // Returns the number of microseconds elapsed in a completed timer.
            virtual uint64 Microseconds() const = 0;

            // Returns the number of nanoseconds elapsed in a completed timer.
            virtual uint64 Nanoseconds() const = 0;

        private:
            DISALLOW_COPY_AND_ASSIGN(TimerInterface);
    };

    class StreamExecutorInterface{
        public:
            StreamExecutorInterface(){}
            // Default destructor for the abstract interface.
            virtual ~StreamExecutorInterface() {}
            // Returns the (transitively) wrapped executor if this executor is
            // wrapping another executor; otherwise, returns this.
            virtual StreamExecutorInterface *GetUnderlyingExecutor() { return this; }
            // See the StreamExecutor interface for comments on the same-named methods.
            virtual Status Init(int device_ordinal,
                    DeviceOptions device_options) = 0;
            virtual Status GetKernel(const MultiKernelLoaderSpec &spec,
                    KernelBase *kernel) {
                return errors::Unimplemented("Not Implemented");
            }
            virtual Status Launch(Stream *stream, const ThreadDim &thread_dims,
                    const BlockDim &block_dims, const KernelBase &k,
                    const KernelArgsArrayBase &args) {
                return errors::Unimplemented("Not Implemented");
            }
            virtual Status BlockHostUntilDone(Stream *stream) = 0;
            // Releases any state associated with the kernel.
            virtual void UnloadKernel(const KernelBase *kernel) {}
            virtual DeviceMemoryBase Allocate(uint64 size, int64 memory_space) = 0;
            DeviceMemoryBase Allocate(uint64 size) {
                return Allocate(size, /*memory_space=*/0);
            }
            virtual void Deallocate(DeviceMemoryBase *mem) = 0;
            virtual bool Memcpy(Stream *stream, void *host_dst,
                    const DeviceMemoryBase &gpu_src, uint64 size) = 0;
            virtual bool Memcpy(Stream *stream, DeviceMemoryBase *gpu_dst,
                    const void *host_src, uint64 size) = 0;

            virtual Status SynchronousMemcpy(DeviceMemoryBase *gpu_dst,
                    const void *host_src, uint64 size) = 0;
            virtual Status SynchronousMemcpy(void *host_dst,
                    const DeviceMemoryBase &gpu_src,
                    uint64 size) = 0;
            virtual Status SynchronousMemcpyDeviceToDevice(
                    DeviceMemoryBase *gpu_dst, const DeviceMemoryBase &gpu_src,
                    uint64 size) = 0;
            virtual bool MemcpyDeviceToDevice(Stream *stream, DeviceMemoryBase *gpu_dst,
                    const DeviceMemoryBase &gpu_src,
                    uint64 size) = 0;
            virtual int PlatformDeviceCount() = 0;
            virtual int64 GetDeviceLoad() { return -1; }
            virtual bool DeviceMemoryUsage(int64 *free, int64 *total) const {
                return false;
            }

            virtual bool AllocateStream(Stream *stream) = 0;
            virtual void DeallocateStream(Stream *stream) = 0;
            virtual bool CreateStreamDependency(Stream *dependent, Stream *other) = 0;
            virtual bool AllocateTimer(Timer *timer) = 0;
            virtual void DeallocateTimer(Timer *timer) = 0;
            virtual bool StartTimer(Stream *stream, Timer *timer) = 0;
            virtual bool StopTimer(Stream *stream, Timer *timer) = 0;

            // Return allocator statistics.
            virtual AllocatorStats* GetAllocatorStats() {
                return nullptr;
            }
            // Creates a new DeviceDescription object. Ownership is transferred to the
            // caller.
            virtual Status CreateDeviceDescription(DeviceDescription* ) const = 0;

            virtual Status GetStatus(Stream *stream) {
                return Status(error::UNIMPLEMENTED, "GetStatus is not supported on this executor.");
            }

            // Each call creates a new instance of the platform-specific implementation of
            // the corresponding interface type.
            virtual std::unique_ptr<EventInterface> CreateEventImplementation() = 0;
            virtual std::unique_ptr<KernelInterface> CreateKernelImplementation() = 0;
            virtual std::unique_ptr<StreamInterface> GetStreamImplementation() = 0;
            virtual std::unique_ptr<TimerInterface> GetTimerImplementation() = 0;

    };
}

#endif
