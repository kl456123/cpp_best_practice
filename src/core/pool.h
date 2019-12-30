#ifndef CORE_POOL_H_
#define CORE_POOL_H_
#include <iostream>
#include <map>
#include <vector>
#include <list>
#include "core/backend.h"
#include "core/port.h"

using namespace std;


/* Note that Pool cannot be template class,
 * it should allocate memory and return void*
 * */

class Pool{
    struct Node{
        int size;
        void* chunk;
    };
    public:
    Pool();
    virtual ~Pool(){Clear();}

    virtual void* Malloc(size_t size, int alignment)=0;

    void* Alloc(int size);

    void Recycle(void* ptr);

    void Clear();

    private:
    std::map<void*, Node*> mAllChunks;
    std::list<Node*> mFreeList;
};




#endif
