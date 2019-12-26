#include "test/test_suite.h"
#include "opencl/OpenCLUtilities/opencl_backend.h"
#include <vector>
#include <memory>
#include <iostream>
#include <CL/cl.hpp>

int ComputeSize(const std::vector<int>& tensor_shape){
    int input_buffer_size = 1;
    for(int i=0;i<tensor_shape.size();i++){
        input_buffer_size*=tensor_shape[i];
    }
    return input_buffer_size;
}


bool AllocateTensorBuffer(OpenclBackend* backend_ptr, std::vector<int>& tensor_shape, cl::Buffer*& tensor_buffer){
    const int input_buffer_size = ComputeSize(tensor_shape);

    cl::Memory* buffer;
    bool is_success = backend_ptr->mAllocateBuffer<float>(input_buffer_size, buffer);
    // cannot use dynamic cast here
    tensor_buffer = (cl::Buffer*)buffer;
    return is_success;
}

void ComputeStride(const std::vector<int>& shape, std::vector<int>& stride){
    stride.resize(shape.size(), 1);
    for(int i=shape.size()-2;i>=0;i--){
        stride[i] = stride[i+1]* shape[i];
    }
}

void ComputeStride(const std::vector<int>& shape, int* stride){
    stride[shape.size()-1] = 1;
    for(int i=shape.size()-2;i>=0;i--){
        stride[i] = stride[i+1]* shape[i];
    }
}

// void ComputeOutputShape(const std::vector<int>& input_shape, const std::vector<int>& output_shape){
// }


class ConvTestCase : public TestCase{
    public:
        virtual bool run(){
            std::unique_ptr<OpenclBackend> backend_ptr(new OpenclBackend());

            // input image
            std::vector<int> input_shape({1,3,224,224});

            cl::Buffer* input_image_buffer;
            cl::Buffer* filter_buffer;
            cl::Buffer* output_buffer;
            cl::Buffer* bias_buffer;
            if(AllocateTensorBuffer(backend_ptr.get(), input_shape, input_image_buffer)){
                std::cout<<"fail to allocate opencl buffer"<<std::endl;
                return false;
            }

            // filter and bias
            // C_in, C_out, K, K
            std::vector<int> bias_shape({3,4, 3,3});
            std::vector<int> filter_shape({4});
            std::vector<int> output_shape({1,4,224,224});
            if(AllocateTensorBuffer(backend_ptr.get(), bias_shape, bias_buffer)){
                return false;
            }
            if(AllocateTensorBuffer(backend_ptr.get(), filter_shape, filter_buffer)){
                return false;
            }
            if(AllocateTensorBuffer(backend_ptr.get(), output_shape, output_buffer)){
                return false;
            }

            cl::Kernel kernel = backend_ptr->runtime_ptr()->BuildKernel("conv_2d", "conv2d_buffer");
            int kernel_size = filter_shape[2];
            int dilation = 1;
            int stride = 1 ;
            int input_stride[4];
            ComputeStride(input_shape, input_stride);
            // nchw
            int output_stride[4];
            ComputeStride(output_shape, input_stride);
            int output_buffer_size = ComputeSize(output_shape);
            // h,w
            int input_spatial_shape[] = {input_shape[2], input_shape[3]};
            kernel.setArg(0, input_image_buffer);
            kernel.setArg(1, filter_buffer);
            kernel.setArg(2, bias_buffer);
            kernel.setArg(3, output_buffer);
            kernel.setArg(4, kernel_size);
            kernel.setArg(5, dilation);
            kernel.setArg(6, stride);
            kernel.setArg(7, input_stride);
            kernel.setArg(8, output_stride);
            kernel.setArg(9, input_spatial_shape);
            backend_ptr->runtime_ptr()->command_queue().enqueueNDRangeKernel(
                    kernel,
                    cl::NullRange,
                    cl::NDRange(output_buffer_size),
                    cl::NullRange
                    );

            // host data
            auto data = std::make_shared<float>(output_buffer_size);
            backend_ptr->mCopyBufferToHost<float>(output_buffer, output_buffer_size, data.get());
            return true;
        }
};

TestSuiteRegister(ConvTestCase, "ConvTestCase");
