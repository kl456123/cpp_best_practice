#include "session/core/device_base.h"
#include "session/utils/logging.h"

DeviceBase::~DeviceBase(){
}

const DeviceAttributes& DeviceBase::attributes() const {
    LOG(FATAL) << "Device does not implement attributes()";
}

const std::string& DeviceBase::name() const {
    LOG(FATAL) << "Device does not implement name()";
}
