#ifndef POOL_H_
#define POOL_H_
#include <iostream>
#include <map>
#include <vector>
#include <list>
#include <memory>


using namespace std;


template<class T>
class Pool{
    struct Node{
        int size;
        shared_ptr<T> chunk;
    };
    public:
        Pool();
        virtual ~Pool(){}

        T* alloc(int size);
        void recycle(T*);
        void clear();
    private:
        std::map<T*, shared_ptr<Node>> mAllChunks;
        std::list<shared_ptr<Node>> mFreeList;
};

#endif
