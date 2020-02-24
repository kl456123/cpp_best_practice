#ifndef STREAM_EXECUTOR_INTERNAL_H_
#define STREAM_EXECUTOR_INTERNAL_H_
#include <string>
#include <vector>
#include <cstdint>
typedef int64_t int64;
using std::string;


// Runtime statistics collected by an allocator. Exactly the same as
// tensorflow::AllocatorStats, but independently defined to preserve the mutual
// independence of StreamExecutor and TensorFlow.
struct AllocatorStats {
  int64 num_allocs;          // Number of allocations.
  int64 bytes_in_use;        // Number of bytes in use.
  int64 peak_bytes_in_use;   // The peak bytes in use.
  int64 largest_alloc_size;  // The largest single allocation seen.

  // The upper limit of bytes of user allocatable device memory, if such a limit
  // is known.
  std::vector<int64> bytes_limit;

  // Stack related memory usage.
  int64 bytes_reserved;       // Number of bytes reserved on the stack.
  int64 peak_bytes_reserved;  // The peak number of bytes reserved on the stack.
  // The upper limit on the number bytes of reservable memory on the stack,
  // if such a limit is known.
  std::vector<int64> bytes_reservable_limit;

  AllocatorStats()
      : num_allocs(0),
        bytes_in_use(0),
        peak_bytes_in_use(0),
        largest_alloc_size(0),
        bytes_reserved(0),
        peak_bytes_reserved(0) {}

  string DebugString() const;
};


#endif
