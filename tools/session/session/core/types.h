#ifndef SESSION_CORE_TYPES_H_
#define SESSION_CORE_TYPES_H_
#include <string>

using std::string;
// A DeviceType is just a string, but we wrap it up in a class to give
// some type checking as we're passing these around
class DeviceType {
    public:
        DeviceType(const char* type)  // NOLINT(runtime/explicit)
            : type_(type) {}

        explicit DeviceType(string type) : type_(type.data(), type.size()) {}

        const char* type() const { return type_.c_str(); }
        const string& type_string() const { return type_; }

        bool operator<(const DeviceType& other) const;
        bool operator==(const DeviceType& other) const;
        bool operator!=(const DeviceType& other) const { return !(*this == other); }

    private:
        string type_;
};
std::ostream& operator<<(std::ostream& os, const DeviceType& d);

const char* const DEVICE_DEFAULT = "DEFAULT";
const char* const DEVICE_CPU = "CPU";
const char* const DEVICE_GPU = "GPU";
const char* const DEVICE_SYCL = "SYCL";

#endif
