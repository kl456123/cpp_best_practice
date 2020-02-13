#ifndef MEMORY_MANAGER_CORE_BFC_ALLOCATOR_H_
#define MEMORY_MANAGER_CORE_BFC_ALLOCATOR_H_
#include <string>
#include <cstdint>
#include <cstddef>

#include "memory_manager/core/allocator.h"


class BFCAllocator : public Allocator {
    public:
        // Takes ownership of sub_allocator.
        BFCAllocator(SubAllocator* sub_allocator, size_t total_memory,
                bool allow_growth, const string& name,
                bool garbage_collection = false);
        ~BFCAllocator() override;

        std::string Name() override { return name_; }

        void* AllocateRaw(size_t alignment, size_t num_bytes) override {
            return AllocateRaw(alignment, num_bytes, AllocationAttributes());
        }

        void* AllocateRaw(size_t alignment, size_t num_bytes,
                const AllocationAttributes& allocation_attr) override;

        void DeallocateRaw(void* ptr) override;

        bool TracksAllocationSizes() const override;

        size_t RequestedSize(const void* ptr) const override;

        size_t AllocatedSize(const void* ptr) const override;

        int64 AllocationId(const void* ptr) const override;

        absl::optional<AllocatorStats> GetStats() override;

        void ClearStats() override;

        void SetTimingCounter(SharedCounter* sc) { timing_counter_ = sc; }

        void SetSafeFrontier(uint64 count) override;

        virtual bool ShouldRecordOpName() const { return false; }

        MemoryDump RecordMemoryMap();

    private:
        struct Bin;

        void* AllocateRawInternal(size_t alignment, size_t num_bytes,
                bool dump_log_on_failure,
                uint64 freed_before_count);

        void* AllocateRawInternalWithRetry(
                size_t alignment, size_t num_bytes,
                const AllocationAttributes& allocation_attr);

        void DeallocateRawInternal(void* ptr);

        // Chunks whose freed_at_count is later than the safe frontier value are kept
        // on a special list and not subject to merging immediately upon being freed.
        //
        // This function sweeps that list looking for Chunks whose timestamp is now
        // safe. When found their freed_at_count is set to 0 and we attempt to merge
        // them with their neighbors.
        //
        // If required_bytes > 0 then this function is being called in the context of
        // a need for this many bytes that could not be satisfied without merging
        // unsafe chunks, so we go ahead and merge the unsafe chunks too, just up to
        // the point that a free chunk of required_bytes is produced.  Note that
        // unsafe merged chunks adopt the most conservative timestamp from their
        // constituents so they're only useful for allocations not requiring a
        // particular timestamp.
        bool MergeTimestampedChunks(size_t required_bytes);

        // A ChunkHandle is an index into the chunks_ vector in BFCAllocator
        // kInvalidChunkHandle means an invalid chunk
        typedef size_t ChunkHandle;
        static const int kInvalidChunkHandle = -1;

        typedef int BinNum;
        static const int kInvalidBinNum = -1;
        // The following means that the largest bin'd chunk size is 256 << 21 = 512MB.
        static const int kNumBins = 21;

        // A Chunk points to a piece of memory that's either entirely free or entirely
        // in use by one user memory allocation.
        //
        // An AllocationRegion's memory is split up into one or more disjoint Chunks,
        // which together cover the whole region without gaps.  Chunks participate in
        // a doubly-linked list, and the prev/next pointers point to the physically
        // adjacent chunks.
        //
        // Since a chunk cannot be partially in use, we may need to split a free chunk
        // in order to service a user allocation.  We always merge adjacent free
        // chunks.
        //
        // Chunks contain information about whether they are in use or whether they
        // are free, and contain a pointer to the bin they are in.
        struct Chunk {
            size_t size = 0;  // Full size of buffer.

            // We sometimes give chunks that are larger than needed to reduce
            // fragmentation.  requested_size keeps track of what the client
            // actually wanted so we can understand whether our splitting
            // strategy is efficient.
            size_t requested_size = 0;

            // allocation_id is set to -1 when the chunk is not in use. It is assigned a
            // value greater than zero before the chunk is returned from
            // AllocateRaw, and this value is unique among values assigned by
            // the parent allocator.
            int64 allocation_id = -1;
            void* ptr = nullptr;  // pointer to granted subbuffer.

            // If not kInvalidChunkHandle, the memory referred to by 'prev' is directly
            // preceding the memory used by this chunk.  E.g., It should start
            // at 'ptr - prev->size'
            ChunkHandle prev = kInvalidChunkHandle;

            // If not kInvalidChunkHandle, the memory referred to by 'next' is directly
            // following the memory used by this chunk.  E.g., It should be at
            // 'ptr + size'
            ChunkHandle next = kInvalidChunkHandle;

            // What bin are we in?
            BinNum bin_num = kInvalidBinNum;

            // Optional count when this chunk was most recently made free.
            uint64 freed_at_count = 0;

            bool in_use() const { return allocation_id != -1; }

#ifdef TENSORFLOW_MEM_DEBUG
            // optional debugging info
            const char* op_name = nullptr;
            uint64 step_id = 0;
            int64 action_count = 0;
#endif

            string DebugString(BFCAllocator* a,
                    bool recurse) NO_THREAD_SAFETY_ANALYSIS {
                string dbg;
                strings::StrAppend(
                        &dbg, "  Size: ", strings::HumanReadableNumBytes(size),
                        " | Requested Size: ", strings::HumanReadableNumBytes(requested_size),
                        " | in_use: ", in_use(), " | bin_num: ", bin_num);
                if (recurse && prev != BFCAllocator::kInvalidChunkHandle) {
                    Chunk* p = a->ChunkFromHandle(prev);
                    strings::StrAppend(&dbg, ", prev: ", p->DebugString(a, false));
                }
                if (recurse && next != BFCAllocator::kInvalidChunkHandle) {
                    Chunk* n = a->ChunkFromHandle(next);
                    strings::StrAppend(&dbg, ", next: ", n->DebugString(a, false));
                }
#ifdef TENSORFLOW_MEM_DEBUG
                strings::StrAppend(&dbg, ", for: ", op_name ? op_name : "UNKNOWN",
                        ", stepid: ", step_id,
                        ", last_action: ", action_count);
#endif
                return dbg;
            }
        };
}

#endif
