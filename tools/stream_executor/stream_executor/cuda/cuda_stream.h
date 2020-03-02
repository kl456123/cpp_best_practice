#ifndef STREAM_EXECUTOR_CORE_CUDA_STREAM_H_
#define STREAM_EXECUTOR_CORE_CUDA_STREAM_H_
#include "stream_executor/core/stream_executor_internal.h"
#include "stream_executor/utils/logging.h"

namespace cuda{
    class CudaExecutor;

    class CudaStream:public internal::StreamInterface{
        public:
            explicit CudaStream(CudaExecutor* parent)
                :parent_(parent), cuda_stream_(nullptr), completed_event_(nullptr){}

            // Note: teardown is handled by a parent's call to DeallocateStream.
            ~CudaStream() override {}

            // Explicitly initialize the CUDA resources associated with this stream, used
            // by StreamExecutor::AllocateStream().
            bool Init();

            // Explicitly destroy the CUDA resources associated with this stream, used by
            // StreamExecutor::DeallocateStream().
            void Destroy();

            // Returns true if no work is pending or executing on the stream.
            bool IsIdle() const;
            // Retrieves an event which indicates that all work enqueued into the stream
            // has completed. Ownership of the event is not transferred to the caller, the
            // event is owned by this stream.
            CudaEventHandle* completed_event() { return &completed_event_; }

            // Returns the GpuStreamHandle value for passing to the CUDA API.
            //
            // Precond: this GpuStream has been allocated (otherwise passing a nullptr
            // into the NVIDIA library causes difficult-to-understand faults).
            CudaStreamHandle cuda_stream() const {
                CHECK(cuda_stream_ != nullptr);
                return const_cast<CudaStreamHandle>(cuda_stream_);
            }
            CudaExecutor* parent() const { return parent_; }
        private:
            CudaExecutor* parent_;
            CudaStreamHandle cuda_stream_;
            // Event that indicates this stream has completed.
            GpuEventHandle completed_event_ = nullptr;
    };

}


#endif
