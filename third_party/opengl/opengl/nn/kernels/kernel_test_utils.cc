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

        void CheckSameTensor(const Tensor* cpu_tensor1, const Tensor* cpu_tensor2){
            auto cpu_shape = cpu_tensor1->shape();
            auto ogl_shape = cpu_tensor2->shape();
            CHECK_EQ(cpu_shape.size(), ogl_shape.size());
            for(int i=0;i<cpu_shape.size();++i){
                EXPECT_EQ(cpu_shape[i], ogl_shape[i]);
            }

            // ogl data
            const float* ogl_output_data = cpu_tensor2->host<float>();
            // original data
            const float* cpu_output_data = cpu_tensor1->host<float>();
            CHECK_EQ(cpu_tensor1->num_elements(), cpu_tensor2->num_elements());
            const int output_num_elements = cpu_tensor1->num_elements();
            for(int i=0;i<output_num_elements;++i){
                EXPECT_EQ(ogl_output_data[i], cpu_output_data[i]);
            }
            // check dtype and dformat
            CHECK_EQ(cpu_tensor1->dformat(), cpu_tensor2->dformat());
            CHECK_EQ(cpu_tensor1->dtype(), cpu_tensor2->dtype());
            CHECK_EQ(cpu_tensor1->mem_type(), cpu_tensor2->mem_type());
        }
    }//namespace testing
}//namespace opengl
