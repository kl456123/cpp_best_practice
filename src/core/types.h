#ifndef CORE_TYPES_H_
#define CORE_TYPES_H_
#include <string>
#include "stream_executor/platform/types.h"

enum class MemoryType{
    DEVICE_MEMORY=0,
    HOST_MEMORY=1
};


// string type wrappered
class DeviceType{
    public:
        explicit DeviceType(std::string type):type_(type.data(),type.size()){}

        //accessor
        //c-style string and c++ string
        const char* type()const{return type_.c_str();}
        const std::string& type_string()const {return type_;}

        bool operator==(const DeviceType& other)const;
        bool operator<(const DeviceType& other)const;
        bool operator!=(const DeviceType& other)const{return !(*this==other);}
    private:
        std::string type_;
};

const char* const DEVICE_DEFAULT="DEFAULT";
const char* const DEVICE_CPU="CPU";
const char* const DEVICE_GPU="GPU";
const char* const DEVICE_SYCL="SYCL";

#endif
