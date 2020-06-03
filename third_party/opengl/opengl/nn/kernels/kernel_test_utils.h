#ifndef OPENGL_NN_KERNELS_KERNEL_TEST_UTILS_H_
#define OPENGL_NN_KERNELS_KERNEL_TEST_UTILS_H_
#include "opengl/core/fbo_session.h"
#include "opengl/core/init.h"
#include "opengl/test/test.h"
#include "opengl/nn/apis/nn_ops.h"
#include "opengl/utils/macros.h"

namespace opengl{
    namespace testing{
        FBOSession* InitSession();
        void InitOGLContext();
    }//namespace testing
}//namespace opengl


#endif
