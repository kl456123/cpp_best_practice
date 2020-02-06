#ifndef MEMORY_MANAGER_CORE_ALLOCATOR_H_
#define MEMORY_MANAGER_CORE_ALLOCATOR_H_
#include <cstdint>
#include <string>

#include "memory_manager/utils/define.h"
#include "memory_manager/utils/logging.h"

// Attributes for a single allocation call. Different calls to the same
// allocator could potentially have different allocation attributes.
struct AllocationAttributes {
    AllocationAttributes() = default;

    AllocationAttributes(bool no_retry_on_failure, bool allocation_will_be_logged)
        : no_retry_on_failure(no_retry_on_failure),
        allocation_will_be_logged(allocation_will_be_logged){}

    // If the first attempt to allocate the memory fails, the allocation
    // should return immediately without retrying.
    // An example use case is optional scratch spaces where a failure
    // has only performance impact.
    bool no_retry_on_failure = false;
    // If a Tensor is allocated without the following set to true, then
    // it is logged as an unknown allocation. During execution Tensors
    // should be allocated through the OpKernelContext which records
    // which Op is performing the allocation, and sets this flag to
    // true.
    bool allocation_will_be_logged = false;

    DISALLOW_COPY_AND_ASSIGN(AllocationAttributes);
};


struct AllocatorStats {
    int64_t num_allocs;          // Number of allocations.
    int64_t bytes_in_use;        // Number of bytes in use.
    int64_t peak_bytes_in_use;   // The peak bytes in use.
    int64_t largest_alloc_size;  // The largest single allocation seen.

    // The upper limit of bytes of user allocatable device memory, if such a limit
    // is known.
    int64_t bytes_limit;

    // Stats for reserved memory usage.
    int64_t bytes_reserved;       // Number of bytes reserved.
    int64_t peak_bytes_reserved;  // The peak number of bytes reserved.
    // The upper limit on the number bytes of reservable memory,
    // if such a limit is known.
    int64_t bytes_reservable_limit;

    AllocatorStats()
        : num_allocs(0),
        bytes_in_use(0),
        peak_bytes_in_use(0),
        largest_alloc_size(0),
        bytes_reserved(0),
        peak_bytes_reserved(0) {}

    std::string DebugString() const;
};

// Allocator is an abstract interface for allocating and deallocating
// device memory.
class Allocator {
    public:
        // Align to 64 byte boundary.
        static constexpr size_t kAllocatorAlignment = 64;

        virtual ~Allocator();

        // Return a string identifying this allocator
        virtual std::string Name() = 0;

        // Return an uninitialized block of memory that is "num_bytes" bytes
        // in size.  The returned pointer is guaranteed to be aligned to a
        // multiple of "alignment" bytes.
        // REQUIRES: "alignment" is a power of 2.
        virtual void* AllocateRaw(size_t alignment, size_t num_bytes) = 0;

        // Return an uninitialized block of memory that is "num_bytes" bytes
        // in size with specified allocation attributes.  The returned pointer is
        // guaranteed to be aligned to a multiple of "alignment" bytes.
        // REQUIRES: "alignment" is a power of 2.
        virtual void* AllocateRaw(size_t alignment, size_t num_bytes,
                const AllocationAttributes& allocation_attr) {
            // The default behavior is to use the implementation without any allocation
            // attributes.
            return AllocateRaw(alignment, num_bytes);
        }

        // Deallocate a block of memory pointer to by "ptr"
        // REQUIRES: "ptr" was previously returned by a call to AllocateRaw
        virtual void DeallocateRaw(void* ptr) = 0;

        // Returns true if this allocator tracks the sizes of allocations.
        // RequestedSize and AllocatedSize must be overridden if
        // TracksAllocationSizes is overridden to return true.
        virtual bool TracksAllocationSizes() const { return false; }

        // Returns true if this allocator allocates an opaque handle rather than the
        // requested number of bytes.
        //
        // This method returns false for most allocators, but may be used by
        // special-case allocators that track tensor usage. If this method returns
        // true, AllocateRaw() should be invoked for all values of `num_bytes`,
        // including 0.
        //
        // NOTE: It is the caller's responsibility to track whether an allocated
        // object is a buffer or an opaque handle. In particular, when this method
        // returns `true`, users of this allocator must not run any constructors or
        // destructors for complex objects, since there is no backing store for the
        // tensor in which to place their outputs.
        virtual bool AllocatesOpaqueHandle() const { return false; }

        // Returns the user-requested size of the data allocated at
        // 'ptr'.  Note that the actual buffer allocated might be larger
        // than requested, but this function returns the size requested by
        // the user.
        //
        // REQUIRES: TracksAllocationSizes() is true.
        //
        // REQUIRES: 'ptr!=nullptr' and points to a buffer previously
        // allocated by this allocator.
        virtual size_t RequestedSize(const void* ptr) const {
            CHECK(false) << "allocator doesn't track sizes";
            return size_t(0);
        }

        // Returns the allocated size of the buffer at 'ptr' if known,
        // otherwise returns RequestedSize(ptr). AllocatedSize(ptr) is
        // guaranteed to be >= RequestedSize(ptr).
        //
        // REQUIRES: TracksAllocationSizes() is true.
        //
        // REQUIRES: 'ptr!=nullptr' and points to a buffer previously
        // allocated by this allocator.
        virtual size_t AllocatedSize(const void* ptr) const {
            return RequestedSize(ptr);
        }

        // Returns either 0 or an identifier assigned to the buffer at 'ptr'
        // when the buffer was returned by AllocateRaw. If non-zero, the
        // identifier differs from every other ID assigned by this
        // allocator.
        //
        // REQUIRES: TracksAllocationSizes() is true.
        //
        // REQUIRES: 'ptr!=nullptr' and points to a buffer previously
        // allocated by this allocator.
        virtual int64_t AllocationId(const void* ptr) const { return 0; }
        // Fills in 'stats' with statistics collected by this allocator.
        virtual AllocatorStats* GetStats() { return nullptr; }
        //
        // Clears the internal stats except for the `in_use` field.
        virtual void ClearStats() {}
};

struct AllocatorAttributes {
    void set_on_host(bool v) { value |= (static_cast<int>(v)); }
    bool on_host() const { return value & 0x1; }
    void set_nic_compatible(bool v) { value |= (static_cast<int>(v) << 1); }
    bool nic_compatible() const { return value & (0x1 << 1); }
    void set_gpu_compatible(bool v) { value |= (static_cast<int>(v) << 2); }
    bool gpu_compatible() const { return value & (0x1 << 2); }

    uint32_t value = 0;

    // Returns a human readable representation of this.
    std::string DebugString() const;
};

// Returns a trivial implementation of Allocator, which is a process singleton.
// Access through this function is only intended for use by restricted parts
// of the infrastructure.
Allocator* cpu_allocator_base();

// If available, calls ProcessState::GetCPUAllocator(numa_node).
// If not, falls back to cpu_allocator_base().
// Intended for use in contexts where ProcessState is not visible at
// compile time. Where ProcessState is visible, it's preferable to
// call it directly.
Allocator* cpu_allocator();

#endif
