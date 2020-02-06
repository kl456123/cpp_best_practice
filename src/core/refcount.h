#ifndef CORE_REFCOUNT_H_
#define CORE_REFCOUNT_H_
#include <atomic>
#include "core/define.h"


class RefCounted{
    public:
        RefCounted();

        void Ref()const;

        bool Unref()const;


        bool RefCountIsOne()const;

    protected:
        virtual ~RefCounted();

    private:
        mutable std::atomic_int_fast32_t ref_;
        DISALLOW_COPY_AND_ASSIGN(RefCounted)

};

inline RefCounted::RefCounted():ref_(1){}
inline RefCounted::~RefCounted(){CHECK_EQ(ref_.load(), 0);}

inline void RefCounted::Ref()const {
    CHECK_GE(ref_.load(), 1);
    ref_.fetch_add(1, std::memory_order_relaxed);
}

inline bool RefCounted::Unref()const{
    CHECK_GT(ref_.load(), 0);
    ref_.fetch_sub(1);
    if(RefCountIsOne()){
        delete this;
        return true;
    }else{
        return false;
    }
}

inline bool RefCounted::RefCountIsOne()const {
    return (ref_.load(std::memory_order_acquire)==1);
}




#endif
