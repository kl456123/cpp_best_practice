#ifndef MEMORY_MANAGER_CORE_ALLOCATOR_REGISTRY_H_
#define MEMORY_MANAGER_CORE_ALLOCATOR_REGISTRY_H_
#include <string>
#include <vector>
#include <memory>

#include "memory_manager/utils/define.h"
#include "memory_manager/core/allocator.h"
/**
 * A singleton registry of AllocatorFactories
 * */
class AllocatorFactory{
    public:
        virtual ~AllocatorFactory(){}

        virtual Allocator* CreateAllocator()=0;
};

class AllocatorFactoryRegistry{
    public:
        AllocatorFactoryRegistry(){}
        ~AllocatorFactoryRegistry(){}
        void Register(const char* source_file,
                int source_line, const std::string& name, int priority,
                AllocatorFactory* factory);
        Allocator* GetAllocator();

        static AllocatorFactoryRegistry* singleton();
    private:
        bool first_alloc_made_ = false;
        struct FactoryEntry{
            const char* source_file;
            int source_line;
            std::string name;
            int priority;
            std::unique_ptr<Allocator> allocator;
            std::unique_ptr<AllocatorFactory> factory;
        };
        std::vector<FactoryEntry> factories_;
        const FactoryEntry* FindEntry(const std::string& name, int priority)const;
        DISALLOW_COPY_AND_ASSIGN(AllocatorFactoryRegistry);
};

class AllocatorFactoryRegistration{
    public:
        AllocatorFactoryRegistration(const char* file, int line, const std::string& name,
                int priority, AllocatorFactory* factory){
            AllocatorFactoryRegistry::singleton()->Register(file, line, name, priority, factory);
        }
};

#define REGISTER_MEM_ALLOCATOR(name, priority, factory)                         \
    REGISTER_MEM_ALLOCATOR_UNIQUE_HELPER(__COUNTER__, __FILE__, __LINE__, name, \
            priority, factory)
#define REGISTER_MEM_ALLOCATOR_UNIQUE_HELPER(ctr, file, line, name, priority, factory)  \
    static AllocatorFactoryRegistration allocator_factory_reg_##ctr(                    \
            file, line, name, priority, new factory)



#endif

