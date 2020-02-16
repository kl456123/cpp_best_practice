#ifndef SESSION_CORE_THREADPOOL_H_
#define SESSION_CORE_THREADPOOL_H_
#include <memory>

#include "session/core/threadpool_options.h"
#include "session/core/threadpool_device.h"
#include "session/utils/macros.h"

namespace thread{
    class ThreadPool{
        public:
            // Constructs a pool that wraps around the thread::ThreadPoolInterface
            // instance provided by the caller. Caller retains ownership of
            // `user_threadpool` and must ensure its lifetime is longer than the
            // ThreadPool instance.
            ThreadPool(ThreadPoolInterface* user_threadpool);

            ~ThreadPool();
        private:
            ThreadPoolInterface* underlying_threadpool_;
            std::unique_ptr<ThreadPoolDevice> threadpool_device_;
            DISALLOW_COPY_AND_ASSIGN(ThreadPool);
    };
}


#endif
