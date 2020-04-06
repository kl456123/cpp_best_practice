#ifndef CORE_MACORS_H_
#define CORE_MACORS_H_

#include "stream_executor/platform/macros.h"
#include "core/logging.h"
#define MEMORY_ALIGN_DEFAULT (1<<5)
#define DELTA 1e-5

#define DLCL_ASSERT(cond)               \
    CHECK(cond)<<"Assert error!\n";


#endif
