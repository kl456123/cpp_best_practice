#include "opengl/utils/env.h"

namespace opengl{
    class PosixEnv : public Env {
        public:
            PosixEnv() {}

            ~PosixEnv() override { LOG(FATAL) << "Env::Default() must not be destroyed"; }
    };

    Env* Env::Default() {
        static Env* default_env = new PosixEnv;
        return default_env;
    }
}
