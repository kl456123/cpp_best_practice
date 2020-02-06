#include "memory_manager/core/allocator.h"
#include "memory_manager/core/allocator_registry.h"
#include "memory_manager/utils/define.h"
#include "memory_manager/core/mem.h"


static bool cpu_allocator_collect_stats = true;


namespace{
    class CPUAllocator:public Allocator{
        public:
            CPUAllocator(){}
            ~CPUAllocator()override{}
            std::string Name()override{return "cpu";}

            void* AllocateRaw(size_t alignment, size_t num_bytes) override {
                void* p = port::AlignedMalloc(num_bytes, alignment);
                if (cpu_allocator_collect_stats) {
                    const std::size_t alloc_size = port::MallocExtension_GetAllocatedSize(p);
                    ++stats_.num_allocs;
                    stats_.bytes_in_use += alloc_size;
                    stats_.peak_bytes_in_use =
                        std::max<int64_t>(stats_.peak_bytes_in_use, stats_.bytes_in_use);
                    stats_.largest_alloc_size =
                        std::max<int64_t>(stats_.largest_alloc_size, alloc_size);
                }
                return p;
            }

            void DeallocateRaw(void* ptr) override {
                if (cpu_allocator_collect_stats) {
                    const std::size_t alloc_size =
                        port::MallocExtension_GetAllocatedSize(ptr);
                    stats_.bytes_in_use -= alloc_size;
                }
                port::AlignedFree(ptr);
            }

            AllocatorStats* GetStats() override {
                return &stats_;
            }

            void ClearStats() override {
                stats_.num_allocs = 0;
                stats_.peak_bytes_in_use = stats_.bytes_in_use;
                stats_.largest_alloc_size = 0;
            }

        private:
            AllocatorStats stats_;
            DISALLOW_COPY_AND_ASSIGN(CPUAllocator);
    };
}

class CPUAllocatorFactory : public AllocatorFactory {
    public:
        Allocator* CreateAllocator() override { return new CPUAllocator; }
};

REGISTER_MEM_ALLOCATOR("DefaultCPUAllocator", 100, CPUAllocatorFactory);
