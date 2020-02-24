#ifndef STREAM_EXECUTOR_PIMPL_H_
#define STREAM_EXECUTOR_PIMPL_H_
#include <memory>
#include <atomic>

#include "stream_executor/core/timer.h"
#include "stream_executor/core/platform.h"
#include "stream_executor/core/device_options.h"
#include "stream_executor/core/stream_executor_internal.h"
#include "stream_executor/core/device_memory_allocator.h"
#include "stream_executor/core/kernel.h"
#include "stream_executor/core/launch_dim.h"
#include "stream_executor/core/stream.h"

// Structure used for device memory leak checking.
struct AllocRecord {
    // The requested allocation size of the buffer.
    uint64 bytes;

    // Holds a representation of the stack at the time the associated buffer was
    // allocated. Produced in a form described in
    // //util/symbolize/symbolized_stacktrace.h.
    string stack_trace;
};

class StreamExecutor{
    public:
        StreamExecutor(const Platform* platform, std::unique_ptr<internal::StreamExecutorInterface> implementation,
                int device_ordinal);

        ~StreamExecutor();

        Status Init();
        Status Init(DeviceOptions device_options);

        const Platform* platform()const {return platform_;}

        // Retrieves (loads) a kernel for the platform this StreamExecutor is acting
        // upon, if one exists.
        //
        // Parameters:
        //   spec: The MultiKernelLoaderSpec is usually generated as a compile-time
        //    constant into an appropriate namespace. For example, see
        //    stream_executor::executor_sample::kKernelLoaderSpecs, from which a
        //    MultiKernelLoaderSpec is selected.
        //   kernel: Outparam that the kernel is loaded into. A given Kernel
        //    instantiation should not be loaded into more than once.
        //
        // If an error occurs, or there is no kernel available for the StreamExecutor
        // platform, error status is returned.
        Status GetKernel(const MultiKernelLoaderSpec &spec, KernelBase *kernel);

        // Releases any state associated with the previously loaded kernel.
        void UnloadKernel(const KernelBase *kernel);

        // Synchronously allocates an array on the device of type T with element_count
        // elements.
        template <typename T>
            DeviceMemory<T> AllocateArray(uint64 element_count, int64 memory_space = 0);

        // As AllocateArray(), but returns a ScopedDeviceMemory<T>.
        template <typename T>
            ScopedDeviceMemory<T> AllocateOwnedArray(uint64 element_count) {
                return ScopedDeviceMemory<T>(this, AllocateArray<T>(element_count));
            }

        // Convenience wrapper that allocates space for a single element of type T in
        // device memory.
        template <typename T>
            DeviceMemory<T> AllocateScalar() {
                return AllocateArray<T>(1);
            }

        // As AllocateScalar(), but returns a ScopedDeviceMemory<T>.
        template <typename T>
            ScopedDeviceMemory<T> AllocateOwnedScalar() {
                return AllocateOwnedArray<T>(1);
            }

        // Deallocate the DeviceMemory previously allocated via this interface.
        // Deallocation of a nullptr-representative value is permitted.
        //
        // Resets the internal contents of mem to be null-representative, but this
        // null-out effect should not be relied upon in client code.
        void Deallocate(DeviceMemoryBase *mem);

        // Returns the device ordinal that this StreamExecutor was initialized with.
        // Meaningless before initialization.
        int device_ordinal() const { return device_ordinal_; }

        // Returns a borrowed pointer to the underlying StreamExecutor implementation.
        internal::StreamExecutorInterface *implementation();
        // Warning: use Stream::ThenLaunch instead, this method is not for general
        // consumption. However, this is the only way to launch a kernel for which
        // the type signature is only known at runtime; say, if an application
        // supports loading/launching kernels with arbitrary type signatures.
        // In this case, the application is expected to know how to do parameter
        // packing that obeys the contract of the underlying platform implementation.
        //
        // Launches a data parallel kernel with the given thread/block
        // dimensionality and already-packed args/sizes to pass to the underlying
        // platform driver.
        //
        // This is called by Stream::Launch() to delegate to the platform's launch
        // implementation in StreamExecutorInterface::Launch().
        Status Launch(Stream *stream, const ThreadDim &thread_dims,
                const BlockDim &block_dims, const KernelBase &kernel,
                const KernelArgsArrayBase &args);
        // Turns StreamExecutor operation tracing on or off.
        void EnableTracing(bool enable);

        // Return allocator statistics.
        AllocatorStats* GetAllocatorStats();
        // Return an allocator which delegates to this stream executor for memory
        // allocation.
        StreamExecutorMemoryAllocator *GetAllocator() { return &allocator_; }

        // Same as SynchronousMemcpy(DeviceMemoryBase*, ...) above.
        Status SynchronousMemcpyH2D(const void *host_src, int64 size,
                DeviceMemoryBase *device_dst);
        // Same as SynchronousMemcpy(void*, ...) above.
        Status SynchronousMemcpyD2H(const DeviceMemoryBase &device_src,
                int64 size, void *host_dst);
    private:
        friend class Stream;
        friend class Timer;

        // Synchronously allocates size bytes on the underlying platform and returns
        // a DeviceMemoryBase representing that allocation. In the case of failure,
        // nullptr is returned.
        DeviceMemoryBase Allocate(uint64 size, int64 memory_space);
        // Causes the host code to synchronously wait for operations entrained onto
        // stream to complete. Effectively a join on the asynchronous device
        // operations enqueued on the stream before this program point.
        Status BlockHostUntilDone(Stream *stream);
        // Without blocking the device, retrieve the current stream status.
        Status GetStatus(Stream *stream);
        // Entrains a memcpy operation onto stream, with a host destination location
        // host_dst and a device memory source, with target size size.
        bool Memcpy(Stream *stream, void *host_dst,
                const DeviceMemoryBase &device_src, uint64 size);

        // Entrains a memcpy operation onto stream, with a device destination location
        // and a host memory source, with target size size.
        bool Memcpy(Stream *stream, DeviceMemoryBase *device_dst,
                const void *host_src, uint64 size);

        // Entrains a memcpy operation onto stream, with a device destination location
        // and a device source location, with target size size. Peer access should
        // have been enabled between the StreamExecutors owning the device memory
        // regions.
        bool MemcpyDeviceToDevice(Stream *stream, DeviceMemoryBase *device_dst,
                const DeviceMemoryBase &device_src, uint64 size);
        // Allocates stream resources on the underlying platform and initializes its
        // internals.
        bool AllocateStream(Stream *stream);

        // Deallocates stream resources on the underlying platform.
        void DeallocateStream(Stream *stream);

        // Allocates timer resources on the underlying platform and initializes its
        // internals.
        bool AllocateTimer(Timer *timer);

        // Deallocates timer resources on the underlying platform.
        void DeallocateTimer(Timer *timer);

        // Records a start event for an interval timer.
        bool StartTimer(Stream *stream, Timer *timer);

        // Records a stop event for an interval timer.
        bool StopTimer(Stream *stream, Timer *timer);

        // Adds an AllocRecord for 'opaque' of size 'bytes' to the record map, for
        // leak checking. NULL buffer pointers and buffer sizes of 0 will not be
        // tracked.
        void CreateAllocRecord(void *opaque, uint64 bytes);

        // Removes the AllocRecord keyed by 'opaque' from the record map. NULL
        // pointers will not be erased (as they're not tracked, per above).
        void EraseAllocRecord(void *opaque);

        // Reference to the platform that created this executor.
        const Platform *platform_;

        // Pointer to the platform-specific-interface implementation. This is
        // delegated to by the interface routines in pointer-to-implementation
        // fashion.
        std::unique_ptr<internal::StreamExecutorInterface> implementation_;

        // A mapping of pointer (to device memory) to string representation of the
        // stack (of the allocating thread) at the time at which the pointer was
        // allocated.
        std::map<void *, AllocRecord> mem_allocs_ ;

        // Slot to cache the owned DeviceDescription for the underlying device
        // once it has been quieried from DeviceDescription().
        mutable std::unique_ptr<DeviceDescription> device_description_;

        // The device ordinal that this object was initialized with.
        //
        // Immutable post-initialization.
        int device_ordinal_;

        // Counter for the current number of live streams. This is used to check
        // for accidentally-outstanding streams at StreamExecutor teardown time, as
        // well
        // as to indicate leaks (via a large outstanding count being logged) in the
        // case we can't allocate more streams.
        std::atomic_int_fast32_t live_stream_count_;

        // Allocated memory in bytes.
        int64 mem_alloc_bytes_;
        // Memory limit in bytes. Value less or equal to 0 indicates there is no
        // limit.
        int64 memory_limit_bytes_;

        StreamExecutorMemoryAllocator allocator_;

        DISALLOW_COPY_AND_ASSIGN(StreamExecutor);


};

template <typename T>
inline DeviceMemory<T> StreamExecutor::AllocateArray(uint64 element_count,
        int64 memory_space) {
    uint64 bytes = sizeof(T) * element_count;
    return DeviceMemory<T>(Allocate(bytes, memory_space));
}

// template implementation
template <typename... Params, typename... Args>
inline Stream &Stream::ThenLaunch(ThreadDim thread_dims, BlockDim block_dims,
        const TypedKernel<Params...> &kernel,
        Args... args) {
    KernelInvocationChecker<std::tuple<Params...>,
    std::tuple<Args...>>::CheckAllStaticAssert();
    if (ok()) {
        // This is the core that allows type-safe kernel launching.
        // Since the platforms take kernel arguments as tuples of (void *, size),
        // we pack the variadic parameters passed as ...args into the desired
        // tuple form and pass that packed form to the StreamExecutor::Launch()
        // implementation.
        KernelArgsArray<sizeof...(args)> kernel_args;
        kernel.PackParams(&kernel_args, args...);
        CHECK(parent_ != nullptr);
        bool ok =
            parent_->Launch(this, thread_dims, block_dims, kernel, kernel_args)
            .ok();
        if (!ok) {
            SetError();
            LOG(WARNING) << "parent failed to launch kernel: " << &kernel;
        }
    }
    return *this;
}

template <typename ElemT>
ScopedDeviceMemory<ElemT>::ScopedDeviceMemory(StreamExecutor *parent,
        DeviceMemoryBase value)
    : wrapped_(value),
    device_ordinal_(parent->device_ordinal()),
    allocator_(parent->GetAllocator()) {}

#endif
