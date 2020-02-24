#ifndef CORE_PLATFORM_H_
#define CORE_PLATFORM_H_
#include "stream_executor/utils/env_time.h"
#include "stream_executor/utils/macros.h"


// An interface used to implementation to access operator system
class Env{
    public:
        Env(){};
        virtual ~Env()=default;
        static Env* Default();

        // file operator(delete or create)

        // time proxy
        virtual uint64_t NowNanos(){return EnvTime::NowNanos();}
        virtual uint64_t NowMicros(){return EnvTime::NowMicros();}
        virtual uint64_t NowSeconds(){return EnvTime::NowSeconds();}
    private:
        DISALLOW_COPY_AND_ASSIGN(Env);
};

#endif
