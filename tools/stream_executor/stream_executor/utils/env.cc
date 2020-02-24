#include "stream_executor/utils/env.h"
#include "stream_executor/utils/logging.h"


// unix interface
class PosixEnv : public Env{
    public:
        PosixEnv(){}
        ~PosixEnv()override{LOG(FATAL)<<"Env::Default() must not be destroyed";}


};

Env* Env::Default(){
    static Env* env_default = new PosixEnv();
    return env_default;
}


