#ifndef ALLOCATOR_H_
#define ALLOCATOR_H_
/***************************
 * class Allocator is just to allocate memory in device or host, dont care of shape or types,
 * shape is cared when create tensor, types is cared in TypedAllocator. AllocationAttributes and
 * AllocatorStatistic can be added here for debugging and collecting information to log.
 * Note that use bytes insteads of number as the argument names
 */
#include <vector>
#include <string>
#include "opengl/core/opengl.h"

namespace opengl{
    struct AllocationAttributes{
    };

    struct AllocatorStatistic{
    };

    class Allocator{
        public:
            virtual std::string Name()=0;
            virtual ~Allocator();
            virtual void* AllocateRaw(size_t num_bytes)=0;
            virtual void* AllocateRaw(size_t num_bytes,
                    const AllocationAttributes& allocation_attr){
                return AllocateRaw(num_bytes);
            }
            virtual void DeAllocateRaw(void* ptr)=0;

            // check if it is opaque or not(note that device memory is refers to opaque memory)
            virtual bool AllocatesOpaqueHandle()const{return false;}

            virtual int AllocationId()const{return 0;}
    };

    // high level api of allocator, used to construct PoolAllocator and BFCAllocator
    class SubAllocator{
        public:
            SubAllocator();
            virtual ~SubAllocator(){}
            virtual void* Alloc(size_t num_bytes)=0;
            virtual void Free(void* ptr)=0;
    };

}//namespace opengl
#endif
