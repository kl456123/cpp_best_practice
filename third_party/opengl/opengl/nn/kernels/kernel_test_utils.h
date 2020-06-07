#ifndef OPENGL_NN_KERNELS_KERNEL_TEST_UTILS_H_
#define OPENGL_NN_KERNELS_KERNEL_TEST_UTILS_H_
#include "opengl/core/fbo_session.h"
#include "opengl/core/init.h"
#include "opengl/test/test.h"
#include "opengl/nn/apis/nn_ops.h"
#include "opengl/utils/macros.h"

namespace opengl{
    namespace testing{
        // initialize environment
        std::unique_ptr<FBOSession> InitSession();
        void InitOGLContext();

        // check utils
        void CheckSameTensor(const Tensor* cpu_tensor1, const Tensor* cpu_tensor2);
        void CheckSameValueTensor(const Tensor* cpu_tensor1, const Tensor* cpu_tensor2);
        void CleanupTensorList(::opengl::TensorList* outputs_tensor);
    }//namespace testing
}//namespace opengl


#endif
