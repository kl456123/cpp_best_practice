#include <vector>
#include <memory>
#include <iostream>
#include <CL/cl.hpp>
#include <stdlib.h>
#include <assert.h>

#include "test_suite.h"
#include "backends/opencl/opencl_backend.h"
#include "core/tensor.h"
#include "test_helper.h"


cl::Buffer OpenCLBuffer(void* device){
    return *((cl::Buffer*)device);
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
        int out_ind,
        int pad
        )
{
    // filter: C_out*K*K*C_in (c_out, k1, k2, c_in)
    // output: N*C_out*H*W (b, c_out, h, w)
    // input: N*C_in*H*W (b, c_in, h+k1-K/2, w+k2-K/2)
    // bias: N*C_out

    // kernel_size = (kernel_size-1) * dilation + 1;
    int out_channels = outputStride[0]/outputStride[1];
    int in_channels = inputStride[0]/inputStride[1];

    int batch_block = outputStride[0];
    int channel_block = outputStride[1];
    int height_block = outputStride[2];
    int width_block = outputStride[3];
    int out_ind_tmp = out_ind;
    int out_b_ind = out_ind_tmp / batch_block;
    out_ind_tmp = out_ind_tmp % batch_block;
    int out_c_ind = out_ind_tmp / channel_block;
    out_ind_tmp = out_ind_tmp % channel_block;
    int out_h_ind = out_ind_tmp / height_block;
    out_ind_tmp = out_ind_tmp % height_block;
    int out_w_ind = out_ind_tmp / width_block;

    int bias_index = out_c_ind+ out_b_ind*out_channels;
    float sum =  bias[bias_index];
    for (int k1 = 0; k1 < kernel_size; k1++)
    {
        for (int k2 = 0; k2 < kernel_size; k2++)
        {
            for (int in_c_ind = 0; in_c_ind < in_channels; in_c_ind++)
            {
                int filter_index = ((out_c_ind * kernel_size + k1) * kernel_size + k2) * in_channels + in_c_ind;
                int in_h_ind = out_h_ind * stride - pad + k1*dilation;
                int in_w_ind = out_w_ind * stride - pad + k2*dilation;
                // check insane
                if (in_h_ind < 0 || in_h_ind >= inputShape[0] || in_w_ind < 0 || in_w_ind >= inputShape[1])
                {
                    continue;
                }
                int input_index = out_b_ind * inputStride[0] + in_c_ind * inputStride[1] +
                    in_h_ind * inputStride[2] + in_w_ind * inputStride[3];
                sum += filter[filter_index] * input[input_index];
            }
        }
    }
    output[out_ind] = sum;
}

float get_random(int max_value){
    return (rand()%int(max_value)+1)/max_value;
}

void ComputeShape(const std::vector<int> input_shape, int dilation, int stride, int pad,int out_channels,
        int kernel_size, std::vector<int>& output_shape){
    output_shape.resize(4);
    output_shape[0] = input_shape[0];
    output_shape[1] = out_channels;
    kernel_size = dilation*(kernel_size-1)+1;

    output_shape[2] = (input_shape[2]-kernel_size + 2*pad)/stride+1;
    output_shape[3] = (input_shape[3]-kernel_size + 2*pad)/stride+1;
}

class ConvTestCase : public TestCase{
    public:
        virtual bool run(){
            // std::unique_ptr<OpenclBackend> backend_ptr(new OpenclBackend());
            // std::unique_ptr<Pool<float>> pool_ptr(new Pool<float>());


            // input image
            int batch_size = 1;
            int input_channels = 3;
            int output_channels = 3;
            int kernel_size = 3;
            int dilation = 2;
            int stride = 2;
            int pad = 2;
            std::vector<int> input_shape({batch_size,input_channels,5,5});
            // filter and bias
            // C_in, C_out, K, K
            std::vector<int> filter_shape({output_channels,input_channels, kernel_size, kernel_size});
            std::vector<int> bias_shape({output_channels});
            std::vector<int> output_shape;
            ComputeShape(input_shape,dilation, stride, pad, output_channels,
                    kernel_size, output_shape);

            shared_ptr<Tensor> input, filter, output, bias, expected_output;

            input.reset(Tensor::Random(input_shape));
            filter.reset(Tensor::Random(filter_shape));
            bias.reset(Tensor::Random(bias_shape));
            output.reset(Tensor::Zeros(output_shape));
            expected_output.reset(Tensor::Zeros(output_shape));

            auto gpu_device = Backend::ForwardType::OPENCL;
            auto backend_ptr = ExtractBackend(gpu_device);

            // copy to gpu
            input->CopyToDevice(gpu_device);
            filter->CopyToDevice(gpu_device);
            bias->CopyToDevice(gpu_device);
            output->CopyToDevice(gpu_device);

            // input->CopyToDevice();

            // Tensor<float> input;
            // Tensor<float> filter;
            // Tensor<float> output;
            // Tensor<float> bias;
            // auto input = shared_ptr<Tensor>();

            // if(!AllocateTensorBuffer(backend_ptr.get(), input_shape, input.buffer)){
            // std::cout<<"fail to allocate opencl buffer"<<std::endl;
            // return false;
            // }


            // if(!AllocateTensorBuffer(backend_ptr.get(), bias_shape, bias.buffer)){
            // return false;
            // }
            // if(!AllocateTensorBuffer(backend_ptr.get(), filter_shape, filter.buffer)){
            // return false;
            // }
            // if(!AllocateTensorBuffer(backend_ptr.get(), output_shape, output.buffer)){
            // return false;
            // }
            // int input_size = ComputeSize(input_shape);
            // int filter_size = ComputeSize(filter_shape);
            // int bias_size = ComputeSize(bias_shape);
            // AllocateTensorHost<float>(pool_ptr.get(), input_shape, input.host);
            // AllocateTensorHost<float>(pool_ptr.get(), bias_shape, bias.host);
            // AllocateTensorHost<float>(pool_ptr.get(), filter_shape, filter.host);
            // AllocateTensorHost<float>(pool_ptr.get(), output_shape, output.host);
            // float max_value = 10000;
            // for(int i=0; i<input_size; i++){
            // input.host[i] = 1;
            // }
            // for(int i=0;i<filter_size; i++){
            // filter.host[i] = 1;
            // }
            // for(int i=0;i<bias_size;i++){
            // bias.host[i] = 1;
            // }

            // // copy to device
            // backend_ptr->mMapHostToBuffer<float>(filter.host,filter_size, filter.buffer);
            // backend_ptr->mMapHostToBuffer<float>(input.host,input_size, input.buffer);
            std::string program_name = "src/backends/opencl/cl/conv_2d.cl";
            std::string kernel_name = "conv2d_buffer";

            cl::Kernel kernel = dynamic_cast<OpenclBackend*>(backend_ptr)->runtime_ptr()->BuildKernel(program_name, kernel_name);


            int input_stride[4];
            int output_stride[4];
            input->stride(input_stride);
            output->stride(output_stride);
            int output_size = output->size();

            // ComputeStride(input_shape, input_stride);
            // nchw
            // ComputeStride(output_shape, output_stride);
            // int output_buffer_size = ComputeSize(output_shape);
            // // h,w
            int input_spatial_shape[] = {input_shape[2], input_shape[3]};
            kernel.setArg(0, OpenCLBuffer(input->device()));
            kernel.setArg(1, OpenCLBuffer(filter->device()));
            kernel.setArg(2, OpenCLBuffer(bias->device()));
            kernel.setArg(3, OpenCLBuffer(output->device()));
            kernel.setArg(4, kernel_size);
            kernel.setArg(5, dilation);
            kernel.setArg(6, stride);
            kernel.setArg(7, pad);
            kernel.setArg(8, input_stride);
            kernel.setArg(9, output_stride);
            kernel.setArg(10, input_spatial_shape);


            dynamic_cast<OpenclBackend*>(backend_ptr)->runtime_ptr()->command_queue().enqueueNDRangeKernel(
                    kernel,
                    cl::NullRange,
                    cl::NDRange(output_size),
                    cl::NullRange
                    );

            output->CopyToHost();

            for(int i=0;i<output_size;i++){
                Conv2dNaive(input->host<float>(), filter->host<float>(), bias->host<float>(),
                        expected_output->host<float>(), kernel_size, dilation, stride,
                        input_stride, output_stride, input_spatial_shape,i, pad);
            }

            expected_output->Print<float>();
            output->Print<float>();

            assert(CompareTensor(expected_output->host<float>(), output->host<float>(), output_size));




            return true;
        }
};

TestSuiteRegister(ConvTestCase, "ConvTestCase");
