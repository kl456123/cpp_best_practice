#include "Pool.h"
#include <limits>


template<class T>
T* Pool<T>::alloc(int size){
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
        T* chunk = (*finalIter)->chunk.get();
        mFreeList.erase(finalIter);
        return chunk;
        // return nullptr;
    }

    // alloc new
    shared_ptr<Node> node(new Node);
    node->chunk.reset(new T[size]);
    node->size = size;
    if(node->chunk==nullptr){
        std::cout<<"Error when alloc"<<std::endl;
        return nullptr;
    }

    mAllChunks.insert(make_pair(node->chunk.get(), node));
    return node->chunk.get();
}


template <typename T>
Pool<T>::Pool(){
}
template<typename T>
void Pool<T>::recycle(T* chunk){
    auto iter = mAllChunks.find(chunk);
    if(iter==mAllChunks.end()){
        std::cout<<"Error "<<std::endl;
        return;
    }
    mFreeList.push_back(iter->second);
}


template<typename T>
void Pool<T>::clear(){
    mFreeList.clear();
    mAllChunks.clear();
}


template class Pool<float>;




