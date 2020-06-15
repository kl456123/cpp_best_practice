#ifndef UTIL_H_
#define UTIL_H_
#include <string>
#include <vector>
#include "opengl/core/types.h"


namespace opengl{
    void setLocalSize(std::vector<std::string>& prefix, int* localSize, std::vector<int> local_sizes);
    IntList AmendShape(const IntList& shape, const int amend_size=4);
}//namespace opengl

#endif
