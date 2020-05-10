#ifndef OPENGL_CORE_TYPES_H_
#define OPENGL_CORE_TYPES_H_
#include <vector>
#include <unordered_map>
#include <functional>

#include "opengl/core/tensor.h"

namespace opengl{
    class Kernel;
    class Context;

    typedef std::vector<Tensor*> TensorList;
    typedef std::vector<std::vector<int>> TensorShapeList;
    typedef std::vector<Kernel*> KernelList;
    typedef std::vector<std::string> StringList;

    typedef std::function<void(Kernel**, Context*)> KernelFactory;
    typedef std::unordered_map<std::string, KernelFactory> KernelMap;

    typedef std::unordered_map<std::string, int> NamedIndex;
    typedef std::vector<std::string> TensorNameList;
    typedef std::vector<std::pair<std::string, Tensor*>> NamedTensorList;
}//namespace opengl


#endif
