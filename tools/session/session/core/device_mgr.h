#ifndef SESSION_CORE_DEVICE_MGR_H_
#define SESSION_CORE_DEVICE_MGR_H_
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "session/core/device.h"
#include "session/utils/status.h"
#include "session/utils/macros.h"

class DeviceAttributes;
using std::string;

// Represents a set of devices.
class DeviceMgr {
    public:
        DeviceMgr() = default;
        virtual ~DeviceMgr();

        // Returns attributes of all devices.
        virtual void ListDeviceAttributes(
                std::vector<DeviceAttributes>* devices) const = 0;

        // Returns raw pointers to the underlying devices.
        virtual std::vector<Device*> ListDevices() const = 0;

        // Returns a string listing all devices.
        virtual string DebugString() const = 0;

        // Returns a string of all the device mapping.
        virtual string DeviceMappingString() const = 0;

        // Assigns *device with pointer to Device of the given name.
        // Accepts either a full device name, or just the replica-local suffix.
        virtual Status LookupDevice(std::string name, Device** device) const = 0;

        // Clears given containers of all devices if 'container' is
        // non-empty. Otherwise, clears default containers of all devices.
        virtual void ClearContainers(std::vector<string> containers) const = 0;

        virtual int NumDeviceType(const string& type) const = 0;

        DISALLOW_COPY_AND_ASSIGN(DeviceMgr);
};

// Represents a static set of devices.
class StaticDeviceMgr : public DeviceMgr {
    public:
        // Constructs a StaticDeviceMgr from a list of devices.
        explicit StaticDeviceMgr(std::vector<std::unique_ptr<Device>> devices);

        // Constructs a StaticDeviceMgr managing a single device.
        explicit StaticDeviceMgr(std::unique_ptr<Device> device);

        ~StaticDeviceMgr() override;

        void ListDeviceAttributes(
                std::vector<DeviceAttributes>* devices) const override;
        std::vector<Device*> ListDevices() const override;
        string DebugString() const override;
        string DeviceMappingString() const override;
        Status LookupDevice(string name, Device** device) const override;
        void ClearContainers(std::vector<string> containers) const override;
        int NumDeviceType(const string& type) const override;

    private:
        const std::vector<std::unique_ptr<Device>> devices_;

        std::string CopyToBackingStore(std::string s);

        std::unordered_map<std::string, Device*> device_map_;
        // core::Arena name_backing_store_;  // Storage for keys in device_map_
        std::unordered_map<string, int> device_type_counts_;

        DISALLOW_COPY_AND_ASSIGN(StaticDeviceMgr);
};

// Represents a dynamic set of devices
class DynamicDeviceMgr : public DeviceMgr {
    public:
        // Constructs an empty DynamicDeviceMgr.
        DynamicDeviceMgr() {}

        ~DynamicDeviceMgr() override;

        void ListDeviceAttributes(
                std::vector<DeviceAttributes>* devices) const override;
        std::vector<Device*> ListDevices() const override;
        string DebugString() const override;
        string DeviceMappingString() const override;
        Status LookupDevice(std::string name, Device** device) const override;
        void ClearContainers(std::vector<string> containers) const override;
        int NumDeviceType(const string& type) const override;

        // Add devices to device manager. Returns error for repeated device names.
        Status AddDevices(std::vector<std::unique_ptr<Device>> devices);

        // Remove devices from device manager. Returns error for non-existing devices.
        Status RemoveDevices(std::vector<Device*> devices);

        // Remove devices from device manager by their names. Returns error for
        // non-existing devices.
        Status RemoveDevicesByName(const std::vector<string>& device_names);

    private:
        std::unordered_map<Device*, std::unique_ptr<Device>> dynamic_devices_;

        std::unordered_map<string, Device*> device_map_ ;

        std::unordered_map<string, int> device_type_counts_;

        DISALLOW_COPY_AND_ASSIGN(DynamicDeviceMgr);
};



#endif
