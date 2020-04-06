#ifndef TEST_HELPER_H_
#define TEST_HELPER_H_
#include <cmath>
#include "core/macros.h"


template<typename T>
bool CompareTensor(T* a, T* b, int size){
    for(int i=0;i<size;i++){
        if(fabs(a[i]-b[i])>DELTA){
            return false;
        }
    }
    return true;
}

template<typename T>
bool CompareTensor(T* a, T b, int size){
    for(int i=0;i<size;i++){
        if(fabs(a[i]-b)>DELTA){
            return false;
        }
    }
    return true;
}


#endif
