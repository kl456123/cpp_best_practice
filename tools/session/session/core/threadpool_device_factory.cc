#include "session/core/device_factory.h"
#include "session/core/threadpool_device.h"
#include "session/utils/status.h"
#include "session/utils/strcat.h"
#include "session/core/allocator.h"

#include <memory>
#include <string>
#include <vector>

class ThreadPoolDeviceFactory :public DeviceFactory{
    public:
        Status ListPhysicalDevices(std::vector<std::string>* devices)override{
            devices->push_back("/physical_device:CPU:0");
            return Status::OK();
        }

        Status CreateDevices(const SessionOptions& options, const std::string& name_prefix,
                std::vector<std::unique_ptr<Device>>* devices)override{
            int n = 1;
            auto iter=options.config.device_count().find("CPU");
            if(iter!=options.config.device_count().end()){
                n = iter->second;
            }
            for(int i=0; i<n; i++){
                std::string name = string_utils::str_cat(name_prefix, "/device:CPU", std::to_string(i));
                std::unique_ptr<ThreadPoolDevice> tpd;
                tpd.reset(new ThreadPoolDevice(options, name , Bytes(256<<20), DeviceLocality(), cpu_allocator()));
                devices->push_back(std::move(tpd));
            }
            return Status::OK();
        }
};

REGISTER_LOCAL_DEVICE_FACTORY("CPU", ThreadPoolDeviceFactory, 60);
