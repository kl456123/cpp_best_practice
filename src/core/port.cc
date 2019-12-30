#include "core/port.h"


namespace port{
    void* AlignMalloc(size_t size, int minimum_alignment){
        int require_alignment = sizeof(void*);
        if(minimum_alignment<require_alignment){
            return Malloc(size);
        }
        void* ptr = nullptr;

        int err = posix_memalign(&ptr, minimum_alignment, size);
        if(err!=0){
            return nullptr;
        }
        return ptr;
    }

    void Free(void* ptr){
        free(ptr);
    }
    void* Malloc(size_t size){
        return malloc(size);
    }

    void AlignFree(void* ptr){
        Free(ptr);
    }
}// port
