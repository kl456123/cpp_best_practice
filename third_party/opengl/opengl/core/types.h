#ifndef OPENGL_CORE_TYPES_H_
#define OPENGL_CORE_TYPES_H_
#include <vector>
#include <unordered_map>
#include <functional>

#include "opengl/core/dlxnet.pb.h"

namespace opengl{
    class Kernel;
    class Context;
    class Tensor;

    typedef std::vector<int> IntList;
    typedef std::vector<Tensor*> TensorList;
    typedef std::vector<IntList> TensorShapeList;
    typedef std::vector<Kernel*> KernelList;
    typedef std::vector<std::unique_ptr<Kernel>>OwnedKernelList;
    typedef std::vector<std::unique_ptr<Tensor>>OwnedTensorList;
    typedef std::vector<std::string> StringList;

    typedef std::function<void(Kernel**, Context*)> KernelFactory;
    typedef std::unordered_map<std::string, KernelFactory> KernelMap;

    typedef std::unordered_map<std::string, int> NamedIndex;
    typedef std::vector<std::string> TensorNameList;
    typedef std::vector<std::pair<std::string, Tensor*>> NamedTensorList;


    // data format
    typedef dlxnet::TensorProto::DataFormat DataFormat;
}//namespace opengl


#endif
