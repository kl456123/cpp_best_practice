#include "session/utils/env.h"
#include "session/utils/logging.h"


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


