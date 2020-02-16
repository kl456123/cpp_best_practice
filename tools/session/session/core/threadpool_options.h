#ifndef SESSION_CORE_THREADPOOL_OPTIONS_H_
#define SESSION_CORE_THREADPOOL_OPTIONS_H_
#include <functional>

namespace thread{
    class ThreadPoolInterface{
        public:
            virtual void Schedule(std::function<void()> fn)=0;
            virtual int NumThreads()const=0;
            virtual ~ThreadPoolInterface(){}
    };


    struct ThreadPoolOptions{
        ThreadPoolInterface* inter_op_threadpool;
        ThreadPoolInterface* intra_op_threadpool;
    };

}

#endif
