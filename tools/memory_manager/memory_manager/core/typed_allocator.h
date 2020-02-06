#ifndef MEMORY_MANAGER_CORE_TYPED_ALLOCATOR_H_
#define MEMORY_MANAGER_CORE_TYPED_ALLOCATOR_H_
#include "memory_manager/core/allocator.h"


class TypedAllocator{
    public:
        // May return NULL if the tensor has too many elements to represent in a
        // single allocation.
        template <typename T>
            static T* Allocate(Allocator* raw_allocator, size_t num_elements,
                    const AllocationAttributes& allocation_attr) {
                // TODO(jeff): Do we need to allow clients to pass in alignment
                // requirements?

                if (num_elements > (std::numeric_limits<size_t>::max() / sizeof(T))) {
                    return nullptr;
                }

                void* p =
                    raw_allocator->AllocateRaw(Allocator::kAllocatorAlignment,
                            sizeof(T) * num_elements, allocation_attr);
                T* typed_p = reinterpret_cast<T*>(p);
                return typed_p;
            }

        template <typename T>
            static void Deallocate(Allocator* raw_allocator, T* ptr,
                    size_t num_elements) {
                if (ptr) {
                    raw_allocator->DeallocateRaw(ptr);
                }
            }

};


#endif
