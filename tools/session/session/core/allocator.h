#ifndef SESSION_CORE_ALLOCATOR_H_
#define SESSION_CORE_ALLOCATOR_H_
#include <string>
#include "session/utils/logging.h"

typedef int32_t int32;
typedef uint32_t uint32;
using std::string;

class Allocator{
};

struct AllocatorAttributes {
  void set_on_host(bool v) { value |= (static_cast<int>(v)); }
  bool on_host() const { return value & 0x1; }
  void set_nic_compatible(bool v) { value |= (static_cast<int>(v) << 1); }
  bool nic_compatible() const { return value & (0x1 << 1); }
  void set_gpu_compatible(bool v) { value |= (static_cast<int>(v) << 2); }
  bool gpu_compatible() const { return value & (0x1 << 2); }
  void Merge(AllocatorAttributes other) {
    value |= other.value;
    if (scope_id != other.scope_id) {
      CHECK(scope_id == 0 || other.scope_id == 0)
          << "At least one scope_id should be zero to merge "
             "AllocatorAttributes but found this.scope_id="
          << scope_id << " and other.scope_id=" << other.scope_id;
      scope_id = scope_id == 0 ? other.scope_id : scope_id;
    }
  }
  // Returns true if the fields set in *this is a subset of or equal to
  // those set in other.
  bool IsEqualOrLessRestrictiveThan(const AllocatorAttributes& other) const {
    return (value | other.value) == other.value;
  }

  // NOTE: The upper 8 bits of the value are reserved for
  // device-specific uses.  Implementors of a device can interpret these
  // upper 8 bits in device-specific ways, and ops implemented for those
  // devices are responsible for setting those 8 bits appropriately.
  uint32 value = 0;
  // EXPERIMENTAL: If this is greater than zero, then allocation is delegated to
  // a named special-purpose allocator on the same device.
  int32 scope_id = 0;

  // Returns a human readable representation of this.
  std::string DebugString() const;
};

inline Allocator* cpu_allocator(){
    static Allocator* alloc = new Allocator();
    return alloc;
}


#endif
