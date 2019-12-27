#include <vector>
#include <memory>
#include <iostream>
#include <CL/cl.hpp>
#include <stdlib.h>

#include "pool/Pool.h"
#include "test/test_suite.h"
#include "opencl/OpenCLUtilities/opencl_backend.h"


template<typename T>
struct Tensor{
    cl::Buffer *buffer;
    T* host;
};

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


template<typename T>
void AllocateTensorHost(Pool<T>* pool_ptr, std::vector<int>& tensor_shape, T*& data_ptr){
    const int input_size = ComputeSize(tensor_shape);
    data_ptr = pool_ptr->alloc(input_size);
}

void ComputeStride(const std::vector<int>& shape, std::vector<int>& stride){
    stride.resize(shape.size(), 1);
    for(int i=shape.size()-2;i>=0;i--){
        stride[i] = stride[i+1]* shape[i+1];
    }
}

void ComputeStride(const std::vector<int>& shape, int* stride){
    stride[shape.size()-1] = 1;
    for(int i=shape.size()-2;i>=0;i--){
        stride[i] = stride[i+1]* shape[i+1];
    }
}


void Conv2dNaive(const float *input,
        const float *filter,
        const float *bias,
        float *output,
        int kernel_size,
        int dilation,
        int stride,
        int* inputStride,
        int* outputStride,
        int* inputShape,
        int out_ind
        )
{
    // filter: C_out*K*K*C_in (c_out, k1, k2, c_in)
    // output: N*C_out*H*W (b, c_out, h, w)
    // input: N*C_in*H*W (b, c_in, h+k1-K/2, w+k2-K/2)
    // bias: N*C_out

    kernel_size = kernel_size * dilation - 1;
    int out_channels = outputStride[0]/outputStride[1];
    int in_channels = inputStride[0]/inputStride[1];

    int batch_block = outputStride[0];
    int channel_block = outputStride[1];
    int height_block = outputStride[2];
    int width_block = outputStride[3];
    int out_b_ind = out_ind / batch_block;
    out_ind = out_ind % batch_block;
    int out_c_ind = out_ind / channel_block;
    out_ind = out_ind % channel_block;
    int out_h_ind = out_ind / height_block;
    out_ind = out_ind % height_block;
    int out_w_ind = out_ind / width_block;
    float sum = 0;
    for (int k1 = 0; k1 < kernel_size; k1++)
    {
        for (int k2 = 0; k2 < kernel_size; k2++)
        {
            for (int in_c_ind = 0; in_c_ind < in_channels; in_c_ind++)
            {
                int filter_index = ((out_c_ind * kernel_size + k1) * kernel_size + k2) * in_channels + in_c_ind;
                int in_h_ind = out_h_ind * stride;
                int in_w_ind = out_w_ind * stride;
                // check insane
                if (in_h_ind < 0 || in_h_ind >= inputShape[0] || in_w_ind < 0 || in_w_ind >= inputShape[1])
                {
                    continue;
                }
                int input_index = out_b_ind * inputStride[0] + in_c_ind * inputStride[1] +
                    in_h_ind * inputStride[2] + in_w_ind * inputStride[3];
                int bias_index = out_c_ind+ out_b_ind*out_channels;
                sum += filter[filter_index] * input[input_index] + bias[bias_index];
            }
        }
    }
    output[out_ind] = sum;
}

class ConvTestCase : public TestCase{
    public:
        virtual bool run(){
            std::unique_ptr<OpenclBackend> backend_ptr(new OpenclBackend());
            std::unique_ptr<Pool<float>> pool_ptr(new Pool<float>());

            // input image
            std::vector<int> input_shape({1,3,224,224});
            // filter and bias
            // C_in, C_out, K, K
            std::vector<int> filter_shape({3,4, 3,3});
            std::vector<int> bias_shape({4});
            std::vector<int> output_shape({1,4,224,224});

            Tensor<float> input;
            Tensor<float> filter;
            Tensor<float> output;
            Tensor<float> bias;

            if(!AllocateTensorBuffer(backend_ptr.get(), input_shape, input.buffer)){
                std::cout<<"fail to allocate opencl buffer"<<std::endl;
                return false;
            }


            if(!AllocateTensorBuffer(backend_ptr.get(), bias_shape, bias.buffer)){
                return false;
            }
            if(!AllocateTensorBuffer(backend_ptr.get(), filter_shape, filter.buffer)){
                return false;
            }
            if(!AllocateTensorBuffer(backend_ptr.get(), output_shape, output.buffer)){
                return false;
            }
            int input_size = ComputeSize(input_shape);
            int filter_size = ComputeSize(filter_shape);
            int bias_size = ComputeSize(bias_shape);
            AllocateTensorHost<float>(pool_ptr.get(), input_shape, input.host);
            AllocateTensorHost<float>(pool_ptr.get(), bias_shape, bias.host);
            AllocateTensorHost<float>(pool_ptr.get(), filter_shape, filter.host);
            AllocateTensorHost<float>(pool_ptr.get(), output_shape, output.host);
            float max_value = 10000;
            for(int i=0; i<input_size; i++){
                input.host[i] = (rand()%int(max_value)+1)/max_value;
            }
            for(int i=0;i<filter_size; i++){
                filter.host[i] = (rand()%int(max_value)+1)/max_value;
            }
            for(int i=0;i<bias_size;i++){
                bias.host[i] = 0;
            }

            // copy to device
            backend_ptr->mMapHostToBuffer<float>(filter.host,filter_size, filter.buffer);
            backend_ptr->mMapHostToBuffer<float>(input.host,input_size, input.buffer);

            cl::Kernel kernel = backend_ptr->runtime_ptr()->BuildKernel("/home/indemind/Documents/Learning/cpp/opencl/cl/conv_2d.cl", "conv2d_buffer");
            int kernel_size = filter_shape[2];
            int dilation = 1;
            int stride = 1 ;
            int input_stride[4];
            ComputeStride(input_shape, input_stride);
            // nchw
            int output_stride[4];
            ComputeStride(output_shape, output_stride);
            int output_buffer_size = ComputeSize(output_shape);
            // h,w
            int input_spatial_shape[] = {input_shape[2], input_shape[3]};
            kernel.setArg(0, *input.buffer);
            kernel.setArg(1, *filter.buffer);
            kernel.setArg(2, *bias.buffer);
            kernel.setArg(3, *output.buffer);
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
            // for(int i=0;i<output_buffer_size;i++){
                // Conv2dNaive(input.host, filter.host, bias.host, output.host, kernel_size, dilation, stride,\
                        // input_stride, output_stride, input_spatial_shape,i);
            // }


            // host data
            // std::shared_ptr<float> data;
            // data.reset(new float[output_buffer_size]);
            backend_ptr->mMapBufferToHost<float>(output.buffer, output_buffer_size, output.host);
            for(int i=0;i<100;i++){
                std::cout<<output.host[i]<<" ";
            }
            std::cout<<std::endl;
            return true;
        }
};

TestSuiteRegister(ConvTestCase, "ConvTestCase");
