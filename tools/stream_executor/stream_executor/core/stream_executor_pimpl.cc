#include <sstream>

#include "stream_executor/core/stream_executor_pimpl.h"
#include "stream_executor/utils/status_macros.h"

namespace{
    bool FLAGS_check_device_leaks = false;
}

namespace{
    string StackTraceIfVLOG10(){
        return "";
    }
}

internal::StreamExecutorInterface *StreamExecutor::implementation() {
    return implementation_->GetUnderlyingExecutor();
}

void StreamExecutor::UnloadKernel(const KernelBase *kernel) {
    implementation_->UnloadKernel(kernel);
}

void StreamExecutor::DeallocateTimer(Timer *timer) {
    return implementation_->DeallocateTimer(timer);
}

Status StreamExecutor::SynchronousMemcpyH2D(
        const void *host_src, int64 size, DeviceMemoryBase *device_dst) {
    LOG(INFO) << "Called StreamExecutor::SynchronousMemcpyH2D(host_src=" << host_src
        << ", size=" << size << ", device_dst=" << device_dst->opaque() << ")";

    Status result;
    // SCOPED_TRACE(TraceListener::SynchronousMemcpyH2D, &result, host_src, size,
    // device_dst);

    result = implementation_->SynchronousMemcpy(device_dst, host_src, size);
    if (!result.ok()) {
        result = errors::Internal(
                "failed to synchronously memcpy host-to-device: host ");
    }

    return result;
}
Status StreamExecutor::SynchronousMemcpyD2H(
        const DeviceMemoryBase &device_src, int64 size, void *host_dst) {
    LOG(INFO) << "Called StreamExecutor::SynchronousMemcpyD2H(device_src="
        << device_src.opaque() << ", size=" << size
        << ", host_dst=" << host_dst << ")";

    Status result;
    // SCOPED_TRACE(TraceListener::SynchronousMemcpyD2H, &result, device_src, size,
    // host_dst);

    result = implementation_->SynchronousMemcpy(host_dst, device_src, size);
    if (!result.ok()) {
        result = errors::Internal(
                "failed to synchronously memcpy device-to-host: device ");
    }

    return result;
}


Status StreamExecutor::GetKernel(const MultiKernelLoaderSpec &spec,
        KernelBase *kernel) {
    return implementation_->GetKernel(spec, kernel);
}

Status StreamExecutor::Launch(Stream *stream,
        const ThreadDim &thread_dims,
        const BlockDim &block_dims,
        const KernelBase &kernel,
        const KernelArgsArrayBase &args) {
    // SubmitTrace(&TraceListener::LaunchSubmit, stream, thread_dims, block_dims,
    // kernel, args);

    return implementation_->Launch(stream, thread_dims, block_dims, kernel, args);
}

bool StreamExecutor::AllocateStream(Stream *stream) {
    live_stream_count_.fetch_add(1, std::memory_order_relaxed);
    if (!implementation_->AllocateStream(stream)) {
        auto count = live_stream_count_.fetch_sub(1);
        CHECK_GE(count, 0) << "live stream count should not dip below zero";
        LOG(INFO) << "failed to allocate stream; live stream count: " << count;
        return false;
    }

    return true;
}

void StreamExecutor::DeallocateStream(Stream *stream) {
    implementation_->DeallocateStream(stream);
    CHECK_GE(live_stream_count_.fetch_sub(1), 0)
        << "live stream count should not dip below zero";
}
void StreamExecutor::Deallocate(DeviceMemoryBase *mem) {
    LOG(INFO) << "Called StreamExecutor::Deallocate(mem=" << mem->opaque()
        << ") mem->size()=" << mem->size() ;

    if (mem->opaque() != nullptr) {
        EraseAllocRecord(mem->opaque());
    }
    implementation_->Deallocate(mem);
    mem->Reset(nullptr, 0);
}

void StreamExecutor::EraseAllocRecord(void *opaque) {
    if (FLAGS_check_device_leaks && opaque != nullptr) {
        if (mem_allocs_.find(opaque) == mem_allocs_.end()) {
            LOG(ERROR) << "Deallocating unknown pointer: " << opaque;
        } else {
            mem_alloc_bytes_ -= mem_allocs_[opaque].bytes;
            mem_allocs_.erase(opaque);
        }
    }
}

Status StreamExecutor::BlockHostUntilDone(Stream *stream) {
    Status result;
    // SCOPED_TRACE(TraceListener::BlockHostUntilDone, &result, stream);

    result = implementation_->BlockHostUntilDone(stream);
    return result;
}


StreamExecutorMemoryAllocator::StreamExecutorMemoryAllocator(
        StreamExecutor *executor)
    : DeviceMemoryAllocator(executor->platform()) {
        stream_executors_ = {executor};
    }

Status StreamExecutorMemoryAllocator::Allocate(
        int device_ordinal, uint64 size, bool retry_on_failure,
        int64 memory_space, OwningDeviceMemory* mem) {
    StreamExecutor * executor=nullptr;
    GetStreamExecutor(device_ordinal, &executor);
    DeviceMemoryBase result = executor->AllocateArray<uint8_t>(size, memory_space);
    if (size > 0 && result == nullptr) {
        return errors::ResourceExhausted(
                "Failed to allocate request for on device ordinal");
    }
    LOG(INFO) <<  "Allocated (B) on device ordinal : ";
    *mem= OwningDeviceMemory(result, device_ordinal, this);
    return Status::OK();
}

DeviceMemoryBase StreamExecutor::Allocate(uint64 size, int64 memory_space) {
    if (memory_limit_bytes_ > 0 &&
            mem_alloc_bytes_ + size > memory_limit_bytes_) {
        LOG(WARNING) << "Not enough memory to allocate " << size << " on device "
            << device_ordinal_
            << " within provided limit. [used=" << mem_alloc_bytes_
            << ", limit=" << memory_limit_bytes_ << "]";
        return DeviceMemoryBase();
    }
    DeviceMemoryBase buf = implementation_->Allocate(size, memory_space);
    LOG(INFO) << "Called StreamExecutor::Allocate(size=" << size
        << ", memory_space=" << memory_space << ") returns " << buf.opaque()
        << StackTraceIfVLOG10();
    CreateAllocRecord(buf.opaque(), size);

    return buf;
}

void StreamExecutor::CreateAllocRecord(void *opaque, uint64 bytes) {
    if (FLAGS_check_device_leaks && opaque != nullptr && bytes != 0) {
        mem_allocs_[opaque] = AllocRecord{bytes, ""};
        mem_alloc_bytes_ += bytes;
    }
}

Status StreamExecutorMemoryAllocator::Deallocate(int device_ordinal,
        DeviceMemoryBase mem) {
    if (!mem.is_null()) {
        StreamExecutor* executor=nullptr;
        GetStreamExecutor(device_ordinal, &executor);
        LOG(INFO) << "Freeing "<< mem.opaque()<<" on device ordinal "<< device_ordinal;
        executor->Deallocate(&mem);
    }
    return Status::OK();
}

Status StreamExecutorMemoryAllocator::GetStreamExecutor(int device_ordinal, StreamExecutor **stream_executor) const {
    if (device_ordinal < 0) {
        stringstream ss("");
        ss<<"device ordinal value ("
            <<device_ordinal <<") must be non-negative";
        return errors::InvalidArgument(ss.str());
    }
    for (StreamExecutor *se : stream_executors_) {
        if (se->device_ordinal() == device_ordinal) {
            *stream_executor = se;
            return Status::OK();
        }
    }
    stringstream ss("");
    ss<<"Device "<<platform()->Name() <<":"
        <<device_ordinal <<" present but not supported";
    return errors::NotFound(ss.str());
}

bool StreamExecutorMemoryAllocator::AllowsAsynchronousDeallocation() const {
    return false;
}

Status StreamExecutorMemoryAllocator::GetStream(
        int device_ordinal, Stream** stream) {
    // CHECK(!AllowsAsynchronousDeallocation())
    // << "The logic below only works for synchronous allocators";
    StreamExecutor * executor=nullptr;
    GetStreamExecutor(device_ordinal, &executor);
    Stream *out = [&] {
        if (!streams_.count(device_ordinal)) {
            auto p = streams_.emplace(std::piecewise_construct,
                    std::forward_as_tuple(device_ordinal),
                    std::forward_as_tuple(executor));
            p.first->second.Init();
            return &p.first->second;
        }
        return &streams_.at(device_ordinal);
    }();
    *stream= out;
    return Status::OK();
}
