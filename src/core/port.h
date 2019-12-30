#ifndef CORE_PORT_H_
#define CORE_PORT_H_
#include <stdlib.h>


// borrow from tensorflow
namespace port{

    void* Malloc(size_t size);

    void* AlignMalloc(size_t size, int minimum_alignment);

    void Free(void* ptr);

    void AlignFree(void* ptr);

}// port


#endif
