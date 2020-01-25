#include "core/platform/env.h"
#include "core/logging.h"


// unix interface
class PosixEnv : public Env{
    public:
        PosixEnv(){}
        ~PosixEnv()override{LOG(FATAL)<<"Env::Default() must not be destroyed";}


};

Env::Default(){
    static Env* env_default = new PosixEnv();
    return env_default;
}


