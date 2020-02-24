#include "stream_executor/core/stream.h"
#include "stream_executor/core/stream_executor_internal.h"
#include "stream_executor/core/stream_executor_pimpl.h"


Stream::Stream(StreamExecutor *parent)
    : parent_(parent),
    implementation_(parent->implementation()->GetStreamImplementation()),
    allocated_(false),
    ok_(false)
      // temporary_memory_manager_(this)
{
    // VLOG_CALL(PARAM(parent));
}



Stream::~Stream() {
    // VLOG_CALL();

    // Ensure the stream is completed.
    auto status = BlockHostUntilDone();
    if (!status.ok()) {
        LOG(WARNING) << "Error blocking host until done in stream destructor: "
            << status;
    }
    // temporary_memory_manager_.ForceDeallocateAll();
    // RunAfterBlockHostUntilDoneCallbacks();

    if (allocated_) {
        parent_->DeallocateStream(this);
    }
}


Stream &Stream::Init() {
    // VLOG_CALL();

    CHECK_EQ(false, allocated_)
        << "stream appears to already have been initialized";
    CHECK(!ok_) << "stream should be in !ok() state pre-initialization";

    if (parent_->AllocateStream(this)) {
        // Successful initialization!
        allocated_ = true;
        ok_ = true;
    } else {
        LOG(ERROR) << "failed to allocate stream during initialization";
    }

    return *this;
}

Status Stream::BlockHostUntilDone() {
    // VLOG_CALL();

    if (!ok()) {
        Status status = errors::Internal(
                "stream did not block host until done; was already in an error state");
        LOG(INFO) << DebugStreamPointers() << " " << status;
        return status;
    }

    // temporary_memory_manager_.DeallocateFinalizedTemporaries();

    Status error = parent_->BlockHostUntilDone(this);
    CheckError(error.ok());

    // RunAfterBlockHostUntilDoneCallbacks();
    return error;
}


string Stream::DebugStreamPointers() const {
    // Relies on the ToVlogString(const void*) overload above.
    return string_utils::str_cat("[stream=", ",impl=", "]");
}
