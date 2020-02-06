#ifndef MEMORY_MANAGER_CORE_MEM_H_
#define MEMORY_MANAGER_CORE_MEM_H_
#include <cstddef>

namespace port{
    // Aligned allocation/deallocation. `minimum_alignment` must be a power of 2
    // and a multiple of sizeof(void*).
    void* AlignedMalloc(size_t size, int minimum_alignment);
    void AlignedFree(void* aligned_memory);

    void* Malloc(size_t size);
    void* Realloc(void* ptr, size_t size);
    void Free(void* ptr);
    // Returns the actual number N of bytes reserved by the malloc for the
    // pointer p.  This number may be equal to or greater than the number
    // of bytes requested when p was allocated.
    //
    // This routine is just useful for statistics collection.  The
    // client must *not* read or write from the extra bytes that are
    // indicated by this call.
    //
    // Example, suppose the client gets memory by calling
    //    p = malloc(10)
    // and GetAllocatedSize(p) may return 16.  The client must only use the
    // first 10 bytes p[0..9], and not attempt to read or write p[10..15].
    //
    // Currently, if a malloc implementation does not support this
    // routine, this routine returns 0.
    std::size_t MallocExtension_GetAllocatedSize(const void* p);
}

#endif
