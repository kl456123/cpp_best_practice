#ifndef OPENGL_CORE_TYPED_ALLOCATOR_H_
#define OPENGL_CORE_TYPED_ALLOCATOR_H_
/************************
 * TypedAllocator is a static member functions collections, so use it directly
 * like `TypedAllocator::Allocate<T>(num_elements);`.
 * note that use num_elements instead of num_bytes
 * This file just contains some template member functions, so dont need source file
 * with it.
 */
#include "opengl/core/allocator.h"

namespace opengl{
    class TypedAllocator{
        public:
            TypedAllocator();
            template<typename T>
                void* Allocate(Allocator* raw_allocator, size_t num_elements){
                }
            template<typename T>
                void Deallocate(){
                }
    };
}

#endif
