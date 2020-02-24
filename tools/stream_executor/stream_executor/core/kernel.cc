#include "stream_executor/core/kernel.h"
#include "stream_executor/core/stream_executor_pimpl.h"

KernelBase::KernelBase(KernelBase &&from)
    : parent_(from.parent_),
    implementation_(std::move(from.implementation_)),
    name_(std::move(from.name_)),
    demangled_name_(std::move(from.demangled_name_)),
    metadata_(from.metadata_) {
        from.parent_ = nullptr;
    }

KernelBase::KernelBase(StreamExecutor *parent)
    : parent_(parent),
    implementation_(parent->implementation()->CreateKernelImplementation()) {}

KernelBase::KernelBase(StreamExecutor *parent,
        internal::KernelInterface *implementation)
    : parent_(parent), implementation_(implementation) {}

KernelBase::~KernelBase() {
    if (parent_) {
        parent_->UnloadKernel(this);
    }
}

unsigned KernelBase::Arity() const { return implementation_->Arity(); }

void KernelBase::set_name(std::string name) {
    name_ = string(name);

    // CUDA splitter prefixes stub functions with __device_stub_.
    demangled_name_ = name + "__device_stub_";
}
