#include "test_suite.h"
#include "opencl_backend.h"
#include <vector>
#include <memory>
#include <iostream>
#include <CL/cl.hpp>


bool AllocateTensorBuffer(OpenclBackend* backend_ptr, std::vector<int>& tensor_shape, cl::Buffer*& buffer){
    int input_buffer_size = 1;
    for(int i=0;i<tensor_shape.size();i++){
        buffer_size*=tensor_shape[i];
    }
    bool is_success = backend_ptr->mAllocateBuffer(input_buffer_size, input_image_buffer);
    buffer = input_image_buffer;
    return is_success;
}


class ConvTestCase : public TestCase{
    public:
        virtual bool run(){
            std::unique_ptr<OpenclBackend> backend_ptr(new OpenclBackend());

            // input image
            std::vector<int> input_shape({1,3,224,224});

            cl::Buffer* input_image_buffer;
            if(AllocateTensorBuffer(backend_ptr.get(), input_shape, input_image_buffer)){
                std::cout<<"fail to allocate opencl buffer"<<std::endl;
                return false;
            }

            // filter and bias
            // C_in, C_out, K, K
            std::vector<int> bias_shape({3,4, 3,3});
            std::vector<int> filter_shape({4});
            if(AllocateTensorBuffer(backend_ptr.get(), bias_shape,bias_buffer )){
                return false;
            }
            if(AllocateTensorBuffer(backend_ptr.get(), filter_shape, filter_buffer)){
                return false;
            }

            auto& kernel = backend_ptr->runtime_ptr()->BuildKernel("conv_2d", "conv2d_buffer");
            kernle.set_args();
        }
};
