
#include <set>
#include <utility>
#include <vector>
#include <algorithm>
#include <string>

#include "session/core/device_set.h"
#include "session/core/device_factory.h"


DeviceSet::DeviceSet() {}

DeviceSet::~DeviceSet() {}

void DeviceSet::AddDevice(Device* device) {
    devices_.push_back(device);
    // for (const string& name :
            // DeviceNameUtils::GetNamesForDeviceMappings(device->parsed_name())) {
        // device_by_name_.insert({name, device});
    // }
    device_by_name_.insert({device->name(), device});
}

Device* DeviceSet::FindDeviceByName(const string& name) const {
    auto iter = device_by_name_.find(name);
    if(iter==device_by_name_.end()){
        return nullptr;
    }
    return iter->second;
}

// static
int DeviceSet::DeviceTypeOrder(const DeviceType& d) {
    return DeviceFactory::DevicePriority(d.type_string());
}

static bool DeviceTypeComparator(const DeviceType& a, const DeviceType& b) {
    // First sort by prioritized device type (higher is preferred) and
    // then by device name (lexicographically).
    auto a_priority = DeviceSet::DeviceTypeOrder(a);
    auto b_priority = DeviceSet::DeviceTypeOrder(b);
    if (a_priority != b_priority) {
        return a_priority > b_priority;
    }

    return std::string(a.type()) < std::string(b.type());
}

std::vector<DeviceType> DeviceSet::PrioritizedDeviceTypeList() const {
    std::vector<DeviceType> result;
    std::set<string> seen;
    for (Device* d : devices_) {
        const auto& t = d->device_type();
        if (seen.insert(t).second) {
            result.emplace_back(t);
        }
    }
    std::sort(result.begin(), result.end(), DeviceTypeComparator);
    return result;
}
