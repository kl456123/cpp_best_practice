#include "stream_executor/utils/env_time.h"
#include <sys/time.h>
#include <time.h>



// get time from std lib
uint64_t EnvTime::NowNanos(){
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (static_cast<uint64_t>(ts.tv_sec) * kSecondsToNanos +
            static_cast<uint64_t>(ts.tv_nsec));
}



