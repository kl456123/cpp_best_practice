#include "opengl/nn/kernels/kernel_test_utils.h"


namespace opengl{
    namespace testing{
        // create a session for all test
        std::unique_ptr<FBOSession> InitSession(){
            auto session = std::unique_ptr<FBOSession>(new FBOSession);
            return session;
        }

        void InitOGLContext(){
            //TODO(breakpoint) how to init once for all test case
            ::opengl::glfw_init();
            ::opengl::glew_init();
        }
    }//namespace testing
}//namespace opengl
