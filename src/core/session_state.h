#ifndef CORE_SESSION_STATE_H_
#define CORE_SESSION_STATE_H_
#include <unordered_map>
#include <string>
#include "core/error.hpp"

class SessionState{
};


class TensorStore{
    public:
        struct TensorAndKey{
            Tensor tensor;
            int64_t id;
            std::string device_name;
            std::string  GetHandle(){
                return std::string();
            }
        };
    Status AddTensor();
    Status SaveTensors();
    private:
        std::unordered_map<std::string, TensorAndKey>
};


#endif
