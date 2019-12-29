#ifndef CORE_POOL_H_
#define CORE_POOL_H_
#include <iostream>
#include <map>
#include <vector>
#include <list>
#include <memory>
#include "core/backend.h"


using namespace std;


class Pool{
    struct Node{
        int size;
        shared_ptr<T> chunk;
    };
    public:
        Pool(Backend::ForwardType type);
        virtual ~CPUBackend(){}

        template<typename T>
        T* Alloc(int size);

        template<typename T>
        void Recycle(T*);

        void Clear();
    private:
        std::map<T*, shared_ptr<Node>> mAllChunks;
        std::list<shared_ptr<Node>> mFreeList;
};

#endif
