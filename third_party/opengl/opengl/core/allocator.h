#ifndef ALLOCATOR_H_
#define ALLOCATOR_H_
#include <vector>
#include "opengl.h"

namespace opengl{
    class Allocator{
        public:
            Allocator();
            void* AllocateRaw(std::vector<size_t>& shape);
            void* AllocateRaw(std::initializer_list<size_t> shape, GLenum texture_format);
            void DeAllocateRaw(void* ptr);
    };
}//namespace opengl
#endif
