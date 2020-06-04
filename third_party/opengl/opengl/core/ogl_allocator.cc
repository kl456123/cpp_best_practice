#include "opengl/core/ogl_allocator.h"
#include "opengl/core/driver.h"
#include "opengl/core/texture.h"
#include "opengl/utils/macros.h"

namespace opengl{
    namespace{
        const int kOGLAlignment = 4;
    }
    OGLTextureAllocator::OGLTextureAllocator()
        :kMaxTextureSize_(GetMaxTextureSize()){}

    void* OGLTextureAllocator::AllocateRaw(size_t alignment, size_t num_bytes){
        CHECK_EQ(alignment, kOGLAlignment);
        const size_t alloc_num_bytes = UP_ROUND(num_bytes, alignment);
        const int image_height = UP_DIV(alloc_num_bytes, kMaxTextureSize_);
        const int image_width = (image_height==1)? alloc_num_bytes: kMaxTextureSize_;

        void* ptr = new Texture({image_width, image_height}, GL_RGBA32F, GL_TEXTURE_2D, nullptr);
        return ptr;
    }

    void OGLTextureAllocator::DeallocateRaw(void* ptr){
        Texture* texture_ptr = reinterpret_cast<Texture*>(ptr);
        delete texture_ptr;
    }

    // stride allocator implementation

    void* StrideAllocator::Allocate(Allocator* raw_allocator, size_t num_elements,
            size_t stride, const AllocationAttributes& allocation_attr){
        // check stride is valid first
        CHECK_EQ(num_elements%stride, 0);

        const size_t aligned_stride = UP_ROUND(stride, kOGLAlignment);
        const size_t requested_num_bytes = aligned_stride * num_elements /stride;
        void* ptr = raw_allocator->AllocateRaw(kOGLAlignment, requested_num_bytes);
        return ptr;
    }

    void StrideAllocator::Deallocate(Allocator* raw_allocator, void* ptr){
        if(ptr){
            raw_allocator->DeallocateRaw(ptr);
        }
    }

    Allocator* ogl_texture_allocator(){
        static Allocator* a = new OGLTextureAllocator();
        return a;
    }


}//namespace opengl
