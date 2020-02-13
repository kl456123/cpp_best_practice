#include "session/core/op_segment.h"
#include "session/utils/errors.h"
#include "session/utils/logging.h"

OpSegment::Item::~Item() {
    for (auto kv : name_kernel) delete kv.second;
}

OpSegment::OpSegment() {}

OpSegment::~OpSegment() {
    for (auto kv : sessions_) delete kv.second;
}

Status OpSegment::FindOrCreate(const string& session_handle,
        const string& node_name, OpKernel** kernel,
        CreateKernelFn create_fn) {
    //TODO(breakpoint) bugs here
    auto session_iter = sessions_.find(session_handle);
    if(session_iter==sessions_.end()){
        return errors::NotFound("Session ", session_handle, " is not found.");
    }
    Item* item = session_iter->second;
    auto kernel_iter = item->name_kernel.find(node_name);
    if(kernel_iter==item->name_kernel.end()){
        return Status::OK();
    }
    *kernel = kernel_iter->second;
    Status s = create_fn(kernel);
    if (!s.ok()) {
        LOG(ERROR) << "Create kernel failed: " << s;
        return s;
    }
    // insert to map
    return Status::OK();
}


void OpSegment::AddHold(const string& session_handle) {
    Item** item = &sessions_[session_handle];
    if (*item == nullptr) {
        *item = new Item;  // num_holds == 1
    } else {
        ++((*item)->num_holds);
    }
}
void OpSegment::RemoveHold(const string& session_handle) {
    Item* item = nullptr;
    {
        auto siter = sessions_.find(session_handle);
        if (siter == sessions_.end()) {
            LOG(INFO) << "Session " << session_handle << " is not found.";
            return;
        }
        item = siter->second;
        if (--(item->num_holds) > 0) {
            return;
        } else {
            sessions_.erase(siter);
        }
    }
    delete item;
}


