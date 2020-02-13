#include "session/core/device_mgr.h"


DeviceMgr::~DeviceMgr() {}

StaticDeviceMgr::StaticDeviceMgr(std::vector<std::unique_ptr<Device>> devices)
    : devices_(std::move(devices)){
        for (auto& d : devices_) {
            // Register under the (1) full name and (2) canonical name.
            // for (const string& name :
            const string& name = d->name();
            device_map_[name] = d.get();
            // }
            // Register under the (3) local name and (4) legacy local name.
            // for (const string& name :
            // DeviceNameUtils::GetLocalNamesForDeviceMappings(d->parsed_name())) {
            // device_map_[CopyToBackingStore(name)] = d.get();
            // }
            device_type_counts_[d->device_type()]++;
    }
}

StaticDeviceMgr::~StaticDeviceMgr() {
    // Release resources ahead of destroying the device manager as the resource
    // destructors (e.g. ~IteratorResource) assume devices still exist.
    for (auto& device : devices_) {
        // device->ClearResourceMgr();
    }
}
void StaticDeviceMgr::ListDeviceAttributes(
        std::vector<DeviceAttributes>* devices) const {
    devices->reserve(devices_.size());
    for (const auto& dev : devices_) {
        devices->emplace_back(dev->attributes());
    }
}

std::vector<Device*> StaticDeviceMgr::ListDevices() const {
    std::vector<Device*> devices(devices_.size());
    for (size_t i = 0; i < devices_.size(); ++i) {
        devices[i] = devices_[i].get();
    }
    return devices;
}

string StaticDeviceMgr::DebugString() const {
    string out;
    for (const auto& dev : devices_) {
        out+=dev->name();
        out+="\n";
    }
    return out;
}

Status StaticDeviceMgr::LookupDevice(std::string name, Device** device) const {
    auto iter = device_map_.find(name);
    if (iter == device_map_.end()) {
        std::vector<std::string> device_names;
        for (auto&& itr : device_map_) {
            device_names.push_back(itr.first);
        }
        LOG(INFO) << "Unknown device: " << name
            << " all devices: " << device_names[0]<< ", ";
        return errors::InvalidArgument(name, " unknown device.");
    }
    *device = iter->second;
    return Status::OK();
}

string StaticDeviceMgr::DeviceMappingString() const {
    string out;
    for (const auto& dev : devices_) {
        if (!dev->attributes().physical_device_desc().empty()) {
            out+= dev->name();
            out+=" -> ";
            out+=dev->attributes().physical_device_desc();
            out+="\n";
        }
    }
    return out;
}
void StaticDeviceMgr::ClearContainers(
        std::vector<string> containers) const {
    Status s;
    // for (const auto& dev : devices_) {
        // if (containers.empty()) {
            // s.Update(dev->resource_manager()->Cleanup(
                        // dev->resource_manager()->default_container()));
        // } else {
            // for (const string& c : containers) {
                // s.Update(dev->resource_manager()->Cleanup(c));
            // }
        // }
        // if (!s.ok()) {
            // LOG(WARNING) << s;
        // }
    // }
}

int StaticDeviceMgr::NumDeviceType(const string& type) const {
    auto iter = device_type_counts_.find(type);
    if (iter != device_type_counts_.end()) return iter->second;
    return 0;
}
