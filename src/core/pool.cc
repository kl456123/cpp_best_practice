#include <limits>
#include "core/pool.h"
#include "core/macros.h"

void* Pool::Alloc(int size){
    auto finalIter = mFreeList.end();
    int minWaste = numeric_limits<int>::max();
    for(auto iterP=mFreeList.begin();iterP!=mFreeList.end();iterP++){
        auto& iter = *iterP;
        if(iter->size>size){
            int waste = iter->size-size;
            if(waste<minWaste){
                finalIter=iterP;
                minWaste = waste;
            }
        }
    }

    if(finalIter!=mFreeList.end()){
        void* chunk = (*finalIter)->chunk;
        mFreeList.erase(finalIter);
        return chunk;
        // return nullptr;
    }

    // alloc new
    Node* node = new Node;

    node->size = size;
    node->chunk = Malloc(size, MEMORY_ALIGN_DEFAULT);

    if(node->chunk==nullptr){
        std::cout<<"Error when alloc"<<std::endl;
        return nullptr;
    }

    mAllChunks.insert(make_pair(node->chunk, node));
    return node->chunk;
}


Pool::Pool(){
}

void Pool::Recycle(void* chunk){
    auto iter = mAllChunks.find(chunk);
    if(iter==mAllChunks.end()){
        std::cout<<"Error "<<std::endl;
        return;
    }
    mFreeList.push_back(iter->second);
}


void Pool::Clear(){
    mFreeList.clear();
    mAllChunks.clear();
}




