#ifndef CORE_ENV_TIME_H_
#define CORE_ENV_TIME_H_
#include <stdint.h>


class EnvTime{
    public:
        static constexpr uint64_t kMicrosToPicos=1000ULL*1000ULL;
        static constexpr uint64_t kMicrosToNanos = 1000ULL;
        static constexpr uint64_t kMillisToMicros = 1000ULL;
        static constexpr uint64_t kMillisToNanos = 1000ULL * 1000ULL;
        static constexpr uint64_t kSecondsToMillis = 1000ULL;
        static constexpr uint64_t kSecondsToMicros = 1000ULL * 1000ULL;
        static constexpr uint64_t kSecondsToNanos = 1000ULL * 1000ULL * 1000ULL;
        EnvTime()=default;
        virtual ~EnvTime()=default;


        static uint64_t NowNanos();

        static uint64_t NowMicros(){return NowNanos()/kMicrosToNanos;}

        static uint64_t NowSeconds(){return NowNanos()/kSecondsToNanos;}

};



#endif
