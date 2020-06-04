#include "opengl/core/allocator.h"
#include "opengl/utils/mem.h"

namespace opengl{
    class CPUAllocator:public Allocator{
        public:
            CPUAllocator(){}

            ~CPUAllocator() override {}

            std::string Name() override { return "cpu"; }

            void* AllocateRaw(size_t alignment, size_t num_bytes) override {
                void* p = port::AlignedMalloc(num_bytes, alignment);
                return p;
            }

            void DeallocateRaw(void* ptr) override {
                port::AlignedFree(ptr);
            }
    };

    Allocator* cpu_allocator(){
        static Allocator* a = new CPUAllocator();
        return a;
    }

}//namespace opengl
