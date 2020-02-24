#ifndef STREAM_EXECUTOR_CORE_LAUNCH_DIM_H_
#define STREAM_EXECUTOR_CORE_LAUNCH_DIM_H_
#include <string>
#include <cstdint>
#include "stream_executor/utils/strcat.h"

using std::string;
typedef uint64_t uint64;


// Basic type that represents a 3-dimensional index space.
struct Dim3D {
    uint64 x, y, z;

    Dim3D(uint64 x, uint64 y, uint64 z) : x(x), y(y), z(z) {}
};

// Thread dimensionality for use in a kernel launch. See file comment for
// details.
struct ThreadDim : public Dim3D {
    explicit ThreadDim(uint64 x = 1, uint64 y = 1, uint64 z = 1)
        : Dim3D(x, y, z) {}

    // Returns a string representation of the thread dimensionality.
    string ToString() const {
        return string_utils::str_cat("ThreadDim{", std::to_string(x), ", ", std::to_string(y), ", ", std::to_string(z), "}");
    }
};

// Block dimensionality for use in a kernel launch. See file comment for
// details.
struct BlockDim : public Dim3D {
    explicit BlockDim(uint64 x = 1, uint64 y = 1, uint64 z = 1)
        : Dim3D(x, y, z) {}

    // Returns a string representation of the block dimensionality.
    string ToString() const {
        return string_utils::str_cat("BlockDim{", std::to_string(x), ", ", std::to_string(y), ", ", std::to_string(z), "}");
    }
};

#endif
