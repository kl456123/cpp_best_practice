#include "session/core/threadpool.h"

namespace thread{
    ThreadPool::ThreadPool(ThreadPoolInterface* user_threadpool) {
        underlying_threadpool_ = user_threadpool;
        // threadpool_device_.reset(new Eigen::ThreadPoolDevice(
        // underlying_threadpool_, underlying_threadpool_->NumThreads(), nullptr));
    }
    ThreadPool::~ThreadPool() {}
}
