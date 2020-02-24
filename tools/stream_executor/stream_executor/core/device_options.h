#ifndef STREAM_EXECUTOR_CORE_DEVICE_OPTIONS_H_
#define STREAM_EXECUTOR_CORE_DEVICE_OPTIONS_H_
#include "stream_executor/utils/logging.h"


class DeviceOptions{
    public:
        // When it is observed that more memory has to be allocated for thread stacks,
        // this flag prevents it from ever being deallocated. Potentially saves
        // thrashing the thread stack memory allocation, but at the potential cost of
        // some memory space.
        static const unsigned kDoNotReclaimStackAllocation = 0x1;
        static const unsigned kMask = 0xf;  // Mask of all available flags.
        // Constructs an or-d together set of device options.
        explicit DeviceOptions(unsigned flags) : flags_(flags) {
            CHECK((flags & kMask) == flags);
        }
        // Factory for the default set of device options.
        static DeviceOptions Default() { return DeviceOptions(0); }
        unsigned flags() const { return flags_; }
        bool operator==(const DeviceOptions& other) const {
            return flags_ == other.flags_;
        }

        bool operator!=(const DeviceOptions& other) const {
            return !(*this == other);
        }

        string ToString() {
            return flags_ == 0 ? "none" : "kDoNotReclaimStackAllocation";
        }
        // Platform-specific device options. Expressed as key-value pairs to avoid
        // DeviceOptions subclass proliferation.
        std::map<string, string> non_portable_tags;
    private:
        unsigned flags_;
};



#endif
