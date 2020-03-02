#ifndef STREAM_EXECUTOR_CUDA_CUDA_TIMER_H_
#define STREAM_EXECUTOR_CUDA_CUDA_TIMER_H_
#include "stream_executor/core/stream_executor_internal.h"
#include "stream_executor/cuda/cuda_types.h"
namespace cuda{

    class CudaExecutor;
    class CudaStream;

    class CudaTimer::public internal::TimerInterface{
        public:
            explicit CudaTimer(CudaExecutor* parent)
                :parent_(parent), start_event_(nullptr), stop_event_(nullptr){}
            ~CudaTimer()override{}

            // Allocates the platform-specific pieces of the timer, called as part of
            // StreamExecutor::AllocateTimer().
            bool Init();

            // Deallocates the platform-specific pieces of the timer, called as part of
            // StreamExecutor::DeallocateTimer().
            void Destroy();
            // Records the "timer start" event at the current point in the stream.
            bool Start(GpuStream* stream);

            // Records the "timer stop" event at the current point in the stream.
            bool Stop(GpuStream* stream);
            // Returns the elapsed time, in milliseconds, between the start and stop
            // events.
            float GetElapsedMilliseconds() const;

            // See Timer::Microseconds().
            // TODO(leary) make this into an error code interface...
            uint64 Microseconds() const override {
                return GetElapsedMilliseconds() * 1e3;
            }
            // See Timer::Nanoseconds().
            uint64 Nanoseconds() const override { return GetElapsedMilliseconds() * 1e6; }
        private:
            CudaExecutor* parent_;
            CudaEventHandle start_event_;
            CudaEventHandle stop_event_;
    };

}


#endif
