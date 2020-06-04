#include "opengl/nn/kernels/kernel_test_utils.h"
#include "opengl/core/ogl_allocator.h"


namespace opengl{
    // cpu allocator
    TEST(AllocatorTest, CPUAllocatorTest){
        Allocator* a = cpu_allocator();
        void* ptr = a->AllocateRaw(10, 1<<5);
        a->DeallocateRaw(ptr);
    }

    TEST(AllocatorTest, CPUStrideAllocatorTest){
        IntList tensor_shape({1, 3, 224, 224});
        int num_elements = 1;
        for(auto item:tensor_shape){
            num_elements*=item;
        }
        // assume float type
        const int num_bytes = num_elements*sizeof(float);
        Allocator* a = cpu_allocator();
        void* ptr = StrideAllocator::Allocate(a,
                num_bytes, 224, AllocationAttributes());
        StrideAllocator::Deallocate(a, ptr);
    }

    // ogl allocator
    TEST(AllocatorTest, OGLAllocatorTest){
    }

    TEST(AllocatorTest, OGLStrideAllocatorTest){
    }
}//namespace opengl
