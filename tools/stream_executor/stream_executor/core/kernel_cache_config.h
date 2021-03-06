#ifndef STREAM_EXECUTOR_CORE_KERNEL_CACHE_CONFIG_H_
#define STREAM_EXECUTOR_CORE_KERNEL_CACHE_CONFIG_H_


// This enum represents potential configurations of L1/shared memory when
// running a particular kernel. These values represent user preference, and
// the runtime is not required to respect these choices.
enum class KernelCacheConfig {
  // Indicates no preference for device L1/shared memory configuration.
  kNoPreference,

  // Indicates a preference for more shared memory than L1 cache.
  kPreferShared,

  // Indicates a preference for more L1 cache than shared memory.
  kPreferL1,

  // Indicates a preference for equal amounts of L1 cache and shared memory.
  kPreferEqual,
};




#endif
