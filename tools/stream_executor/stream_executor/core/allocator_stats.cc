#include "stream_executor/core/allocator_stats.h"

string AllocatorStats::DebugString() const {
    return
        string("Limit:        ")+"\nInUse:        "+std::to_string(this->bytes_in_use)+
        "\nMaxInUse:     "+std::to_string(this->peak_bytes_in_use)+
        "\nNumAllocs:    "+std::to_string(this->num_allocs)+
        "\nMaxAllocSize: "+std::to_string(this->largest_alloc_size);
}
