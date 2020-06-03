#include <cmath>
#include <memory>
#include <random>
#include "opengl/nn/kernels/kernel_test_utils.h"
#include "opengl/nn/kernels/conv2d.h"

using namespace ::opengl::testing;

namespace opengl{
    namespace{
        // conv2d params
        int input_width = 3;
        int input_height = 3;
        int input_channels = 3;
        int output_channels = 1;
        int num_inputs = 1;
        int kernel_size = 3;
        int stride = 1;
        int padding = 1;
        int groups = 1;
        int dilation = 1;
        bool use_bias = true;



        // some params
        constexpr int num_iters = 10;
        constexpr float precision = 1e-4;

        // cpu version of conv2d used for check the correctness
        void Conv2DCPU(const float* input_data,
                const float* filter_data,
                const float* bias_data,
                float* output_data,
                int kernel_size,
                int stride,
                int padding,
                int input_width,
                int input_height,
                int output_width,
                int output_height,
                int input_channels,
                int output_channels,
                int dilation,
                int groups){
            // input_data: (N, H, W, C)
            // output_data: (N, H, W, C)
            // filter_data: (N_out, N_in, H, W)
            for(int oc=0;oc<output_channels;++oc){
                const int filter_base = oc*kernel_size*kernel_size*input_channels;
                for(int i=0;i<output_height;++i){
                    for(int j=0;j<output_width;++j){
                        int output_index = i*output_width+j;
                        float sum = bias_data==nullptr? 0:bias_data[oc];
                        for(int r=0;r<kernel_size;++r){
                            for(int s=0;s<kernel_size;++s){
                                int input_index_x = j*stride-padding+s*dilation;
                                int input_index_y = i*stride-padding+r*dilation;
                                int input_index = input_index_y*input_width+input_index_x;
                                if(input_index_x<0||input_index_x>=input_width){
                                    continue;
                                }

                                if(input_index_y<0||input_index_y>=input_height){
                                    continue;
                                }
                                int filter_index=r*kernel_size+s;
                                for(int c=0;c<input_channels;++c){
                                    float a = input_data[input_index*input_channels+c];
                                    float b = filter_data[filter_base+c*kernel_size*kernel_size+filter_index];
                                    sum+=a*b;
                                }
                            }
                        }
                        output_data[output_index*output_channels+oc] = sum;
                    }
                }
            }
        }




        const ::dlxnet::ModelProto BuildGraph(){
            auto scope = std::unique_ptr<Scope>(new Scope());
            auto scope_ptr = scope.get();

            int input_id = AddInputNode(scope_ptr, "input");

            // weight
            int weight_id = AddConstNode(scope_ptr, "weight", {output_channels, input_channels,
                    kernel_size,kernel_size}, dlxnet::TensorProto::HWN4C4);
            std::vector<int> input_ids({input_id, weight_id});
            if(use_bias){
                // bias
                int bias_id = AddConstNode(scope_ptr, "bias",
                        {1, output_channels, 1, 1}, dlxnet::TensorProto::NHWC4);
                input_ids.emplace_back(bias_id);
            }

            // add conv node
            AddConvNode(scope_ptr, "output", input_ids,
                    {kernel_size, stride, padding});

            // add meta data
            scope->AddOutputName("output");
            return scope->ExportModel();
        }


        void SingleInference(){
            auto session = InitSession();

            session->LoadGraph(BuildGraph());

            std::vector<int> image_shape = {num_inputs, input_height, input_width, input_channels};
            ::opengl::NamedTensorList inputs(1);
            ::opengl::TensorList outputs_cpu;
            inputs[0].first = "input";
            inputs[0].second = Tensor::Random(Tensor::DT_FLOAT, image_shape);

            ::opengl::TensorNameList output_names({"output", "weight"});
            ::opengl::StringList dformats({"NHWC", "NCHW"});
            if(use_bias){
                output_names.emplace_back("bias");
                dformats.emplace_back("NHWC");
            }
            session->Setup(inputs);

            // do computation for the graph
            session->Run();

            // get cpu outputs from device
            session->GetOutputs(output_names, dformats, &outputs_cpu);

            const float* ogl_output_data = outputs_cpu[0]->host<float>();
            const int output_num_elements = outputs_cpu[0]->num_elements();
            const float* ogl_filter_data = outputs_cpu[1]->host<float>();
            const int filter_num_elements = outputs_cpu[1]->num_elements();
            const float* ogl_bias_data = nullptr;
            if(use_bias){
                ogl_bias_data = outputs_cpu[2]->host<float>();
            }

            // nhwc
            auto output_shape = outputs_cpu[0]->shape();
            const int output_width = output_shape[2];
            const int output_height = output_shape[1];
            const int output_channels = output_shape[3];

            Tensor *cpu_output_tensor  = Tensor::Empty(Tensor::DT_FLOAT, output_shape);
            const float* cpu_input_data = inputs[0].second->host<float>();
            float* cpu_output_data = cpu_output_tensor->host<float>();
            const float* cpu_bias_data = ogl_bias_data;
            const float* cpu_filter_data = ogl_filter_data;

            // compute in cpu
            Conv2DCPU(cpu_input_data, cpu_filter_data, cpu_bias_data, cpu_output_data,
                    kernel_size, stride, padding, input_width, input_height,
                    output_width, output_height, input_channels, output_channels, dilation, groups);

            for(int i=0;i<output_num_elements;++i){
                float actual_value = ogl_output_data[i];
                float expect_value = cpu_output_data[i];
                EXPECT_TRUE(std::fabs(actual_value - expect_value)< precision)<<"Error When index: "<< i
                    <<" Actualy Value: "<<actual_value<<" Extect Value: "<<expect_value;
            }
        }
    }//namespace

    TEST(Conv2dTest, DifferentInputShape){
        InitOGLContext();

        // loop input shape
        for(int size=1;size<=256;size*=2){
            for(int channel=1;channel<=20;channel++){
                input_channels = channel;
                // set conv2d params first
                // const int size = 2;
                LOG(INFO)<<"size: "<<size;
                input_height = size;
                input_width = size;

                SingleInference();
            }
        }
    }

    TEST(Conv2dTest, WithoutBias){
        InitOGLContext();

        // loop input shape
        for(int size=1;size<=256;size*=2){
            // set conv2d params first
            // const int size = 2;
            LOG(INFO)<<"size: "<<size;
            input_height = size;
            input_width = size;
            use_bias = false;

            SingleInference();
        }
    }

    TEST(Conv2dTest, DifferentKernelShape){
        InitOGLContext();

        // loop input shape
        // for(int size=1;size<=256;size*=2){
        // set conv2d params first
        // const int size = 2;
        input_channels = 10;
        output_channels = 5;
        const int size = 3;
        LOG(INFO)<<"size: "<<size;
        input_height = size;
        input_width = size;
        use_bias = false;

        SingleInference();
        // }
    }
}//namespace
