#include <cstdint>

#include "stream_executor/core/stream.h"
#include "stream_executor/core/device_memory.h"
#include "stream_executor/utils/status.h"
#include "stream_executor/utils/macros.h"
#include "stream_executor/utils/errors.h"


class DeviceMemoryAllocator;
class StreamExecutor;


template<typename ElemT>
class ScopedDeviceMemory{
    public:
        ScopedDeviceMemory():device_ordinal_(-1), allocator_(nullptr){}

        // Construct a ScopedDeviceMemory from a custom allocator.
        //
        // Parameters:
        //  mem: Already-allocated device memory value for this scoped mechanism to
        //       deallocate. This memory must have been allocated by parent.
        //  device_ordinal: Device on which the memory was allocated.
        //  allocator: Allocator used to deallocate memory when this instance goes
        //             out of scope.
        ScopedDeviceMemory(DeviceMemoryBase mem, int device_ordinal,
                DeviceMemoryAllocator *allocator)
            : wrapped_(mem), device_ordinal_(device_ordinal), allocator_(allocator) {}

        // A helper constructor to generate a scoped device memory given an already
        // allocated memory and a stream executor.
        //
        // Precondition: memory was allocated by the stream executor `parent`.
        ScopedDeviceMemory(StreamExecutor *parent, DeviceMemoryBase value);

        // Moves ownership of the memory from other to the constructed
        // object.
        //
        // Postcondition: other == nullptr.
        ScopedDeviceMemory(ScopedDeviceMemory &&other)
            : ScopedDeviceMemory(other.Release(), other.device_ordinal_,
                    other.allocator_) {}
        // Releases the memory that was provided in the constructor, through the
        // "parent" StreamExecutor.
        ~ScopedDeviceMemory() { CHECK_OK(Free()); }
        // Moves ownership of the memory from other to this object.
        //
        // Postcondition: other == nullptr.
        ScopedDeviceMemory &operator=(ScopedDeviceMemory &&other) {
            CHECK_OK(Free());
            wrapped_ = other.Release();
            allocator_ = other.allocator_;
            device_ordinal_ = other.device_ordinal_;
            return *this;
        }
        // Returns the memory that backs this scoped allocation converted to
        // DeviceMemory<T> apparent type. This is useful for cases where the
        // DeviceMemory must be passed by const-ref, as the ScopedDeviceMemory doesn't
        // allow copying, for scoped-object-lifetime reasons.
        const DeviceMemory<ElemT> &cref() const { return wrapped_; }

        // Returns a pointer to the DeviceMemory<T> apparent type for use in mutable
        // operations. The value returned should not be used outside the scope of this
        // ScopedDeviceMemory object's lifetime.
        DeviceMemory<ElemT> *ptr() { return &wrapped_; }
        const DeviceMemory<ElemT> *ptr() const { return &wrapped_; }
        // Smart-pointer-like operators for the wrapped DeviceMemory.
        // This reference must not be used outside the lifetime of this
        // ScopedDeviceMemory.
        const DeviceMemory<ElemT> &operator*() const { return cref(); }
        bool is_null() const { return wrapped_.is_null(); }
        bool operator==(std::nullptr_t other) const { return is_null(); }
        bool operator!=(std::nullptr_t other) const { return !is_null(); }
        // Analogous to std::unique_ptr::release, releases ownership of the held
        // memory and transfers it to the caller.
        //
        // Postcondition: *this == nullptr
        DeviceMemory<ElemT> Release() {
            DeviceMemory<ElemT> tmp = wrapped_;
            wrapped_ = DeviceMemory<ElemT>{};
            return tmp;
        }
        // The returned allocator is nonnull iff this object is active.
        DeviceMemoryAllocator *allocator() const { return allocator_; }

        int device_ordinal() const { return device_ordinal_; }

        // Frees the existing memory, resets the wrapped memory to null.
        Status Free();


    private:
        DeviceMemory<ElemT> wrapped_;       // Value we wrap with scoped-release.
        int device_ordinal_;                // Negative one for inactive object.
        DeviceMemoryAllocator *allocator_;  // Null if this object is inactive.
        DISALLOW_COPY_AND_ASSIGN(ScopedDeviceMemory);
};

// Type alias for compatibility with the previous managed memory implementation.
using OwningDeviceMemory = ScopedDeviceMemory<uint8_t>;


class DeviceMemoryAllocator{
    public:
        // Parameter platform indicates which platform the allocator allocates memory
        // on. Must be non-null.
        explicit DeviceMemoryAllocator(const Platform* platform)
            : platform_(platform) {}
        virtual ~DeviceMemoryAllocator() {}

        virtual Status Allocate(int device_ordinal,
                uint64_t size, bool retry_on_failure, int64_t memory_space, OwningDeviceMemory* )=0;
        // Two-arg version of Allocate(), which sets retry-on-failure to true and
        // memory_space to default (0).
        //
        // (We don't simply use a default argument on the virtual Allocate function
        // because default args on virtual functions are disallowed by the Google
        // style guide.)
        Status Allocate(int device_ordinal, uint64 size, OwningDeviceMemory* mem) {
            return Allocate(device_ordinal, size, /*retry_on_failure=*/true,
                    /*memory_space=*/0, mem);
        }

        // Typed version of the allocation, returning typed memory.
        template <typename ElemT>
            Status Allocate(
                    int device_ordinal, uint64_t size, bool retry_on_failure = true,
                    int64 memory_space = 0, ScopedDeviceMemory<ElemT>* mem=nullptr) {
                return Allocate(device_ordinal, size, retry_on_failure, memory_space, mem);
            }
        // Must be a nop for null pointers. Should not be used.
        //
        // TODO(cheshire): Add deprecation notice.
        virtual Status Deallocate(int device_ordinal, DeviceMemoryBase mem) = 0;

        // Return the platform that the allocator allocates memory on.
        const Platform* platform() const { return platform_; }

        // Returns a stream pointer on which it is always safe to access memory
        // allocated by this allocator. It is not necessary to use the returned stream
        // though, as clients may have additional information letting them safely use
        // a different stream.
        virtual Status GetStream(int device_ordinal, Stream**) = 0;
        // Can we call Deallocate() as soon as a computation has been scheduled on
    // a stream, or do we have to wait for the computation to complete first?
    virtual bool AllowsAsynchronousDeallocation() const { return false; }
    protected:
        const Platform* platform_;
};


// Default memory allocator for a platform which uses
// StreamExecutor::Allocate/Deallocate.
class StreamExecutorMemoryAllocator : public DeviceMemoryAllocator {
    // Create an allocator supporting a single device, corresponding to the passed
    // executor.
    explicit StreamExecutorMemoryAllocator(StreamExecutor *executor);
    Status Allocate(int device_ordinal, uint64 size,
            bool retry_on_failure,
            int64 memory_space, OwningDeviceMemory*) override;
    Status Deallocate(int device_ordinal, DeviceMemoryBase mem) override;
    // Gets-or-creates a stream for a given `device_ordinal` from an appropriate
    // stream executor.
    Status GetStream(int device_ordinal, Stream**) override;

    // Gets the stream executor for given device ordinal.
    Status GetStreamExecutor(int device_ordinal, StreamExecutor**) const;

    bool AllowsAsynchronousDeallocation() const override;
    private:
    // Available stream executors. Each stream executor has a different device
    // ordinal.
    std::vector<StreamExecutor *> stream_executors_;

    // Cache of streams for GetStream.
    std::map<int, Stream> streams_ ;
};


template <typename ElemT>
Status ScopedDeviceMemory<ElemT>::Free() {
    if (!wrapped_.is_null()) {
        CHECK(allocator_ != nullptr) << "Owning pointer in inconsistent state";
        RETURN_IF_ERROR(allocator_->Deallocate(device_ordinal_, wrapped_));
    }
    wrapped_ = DeviceMemory<ElemT>{};
    return Status::OK();
}


