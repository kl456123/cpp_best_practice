#include <sstream>
#include <string>
#include <stdio.h>

#include "memory_manager/core/allocator.h"
#include "memory_manager/core/allocator_registry.h"


std::string AllocatorStats::DebugString() const {
    char str[200];
    sprintf(str, "Limit:        %20lld\nInUse:        %20lld\nMaxInUse:     %20lld\nNumAllocs:    %20lld\nMaxAllocSize: %20lld\n",
    static_cast<long long>(0),
    static_cast<long long>(this->bytes_in_use),
    static_cast<long long>(this->peak_bytes_in_use),
    static_cast<long long>(this->num_allocs),
    static_cast<long long>(this->largest_alloc_size));
    return std::string(str);
}

constexpr size_t Allocator::kAllocatorAlignment;

Allocator::~Allocator() {}

std::string AllocatorAttributes::DebugString() const {
    std::stringstream ss("AllocatorAttributes(on_host=");
    ss<<on_host();
    ss<< " nic_compatible=";
    ss<<nic_compatible();
    ss<<" gpu_compatible=";
    ss<<gpu_compatible();
    ss<<")";
    return ss.str();
}

Allocator* cpu_allocator_base() {
    static Allocator* cpu_alloc =
        AllocatorFactoryRegistry::singleton()->GetAllocator();
    // TODO(tucker): This really seems wrong.  It's only going to be effective on
    // the first call in a process (but the desired effect is associated with a
    // session), and we probably ought to be tracking the highest level Allocator,
    // not the lowest.  Revisit the advertised semantics of the triggering option.
    return cpu_alloc;
}

Allocator* cpu_allocator() {
    // Correctness relies on devices being created prior to the first call
    // to cpu_allocator, if devices are ever to be created in the process.
    // Device creation in turn triggers ProcessState creation and the availability
    // of the correct access pointer via this function call.
    return cpu_allocator_base();
}


