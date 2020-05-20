#include <cmath>
#include <memory>
#include <random>
#include "opengl/core/fbo_session.h"
#include "opengl/core/init.h"
#include "opengl/core/scope.h"
#include "opengl/test/test.h"
#include "opengl/utils/macros.h"
#include "opengl/nn/kernels/conv2d.h"

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

        struct Conv2dParams{
            int kernel_size;
            int stride;
            int padding;
        };

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

        void InitOGLContext(){
            //TODO(breakpoint) how to init once for all test case
            ::opengl::glfw_init();
            ::opengl::glew_init();
        }

        int AddConstNode(Scope* scope, const std::string&  name, const std::vector<int>& shape){
            auto node_ptr = scope->AddNode();
            node_ptr->set_name(name);
            node_ptr->set_type("Const");
            int tensor_id = scope->AddTensor(name);
            node_ptr->add_output_index(tensor_id);
            dlxnet::TensorProto* dlcl_tensor = node_ptr->mutable_attr()
                ->mutable_const_attr()->mutable_value();

            int num_elements = 1;
            for(auto dim: shape){
                dlcl_tensor->add_dims(dim);
                num_elements  *= dim;
            }
            for(int j=0;j<num_elements;++j){
                dlcl_tensor->add_float_data(1.0*random()/RAND_MAX);
            }
            // set tensor
            dlcl_tensor->set_data_type(dlxnet::TensorProto::FLOAT32);
            if(name=="weight"){
                dlcl_tensor->set_target_data_format(dlxnet::TensorProto::HWN4C4);
            }else{
                dlcl_tensor->set_target_data_format(dlxnet::TensorProto::NHWC4);
            }
            dlcl_tensor->set_data_format(dlxnet::TensorProto::NCHW);
            return tensor_id;

        }

        // TODO(breakpoint) change id to NodeOut class to store shape info
        // can use shape info do some things to validate
        int AddConvNode(Scope* scope, const std::string&  name, std::vector<int> input_ids,
                const Conv2dParams& conv2d_params){
            auto dlcl_node = scope->AddNode();
            // set node name with the name of the first output
            dlcl_node->set_name(name);

            dlcl_node->set_type("Conv");
            dlxnet::Conv2dAttribute* dst_attr = dlcl_node->mutable_attr()->mutable_conv2d_attr();
            for(int i=0;i<2;++i){
                dst_attr->add_kernel_shape(conv2d_params.kernel_size);
            }
            for(int i=0;i<2;++i){
                dst_attr->add_strides(conv2d_params.stride);
            }
            for(int i=0;i<4;++i){
                dst_attr->add_pads(conv2d_params.padding);
            }
            for(auto tensor_id:input_ids){
                dlcl_node->add_input_index(tensor_id);
            }
            // both node name and tensor name are the same
            int tensor_id = scope->AddTensor(name);
            dlcl_node->add_output_index(tensor_id);
            return tensor_id;
        }

        int AddInputNode(Scope* scope, std::string name){
            // add node
            scope->AddInputName(name);
            // add tensor
            return scope->AddTensor(name);
        }

        const ::dlxnet::ModelProto BuildGraph(){
            auto scope = std::unique_ptr<Scope>(new Scope());
            auto scope_ptr = scope.get();

            int input_id = AddInputNode(scope_ptr, "input");

            // weight
            int weight_id = AddConstNode(scope_ptr, "weight", {output_channels, input_channels,
                    kernel_size,kernel_size});
            std::vector<int> input_ids({input_id, weight_id});
            if(use_bias){
                // bias
                int bias_id = AddConstNode(scope_ptr, "bias", {1, output_channels, 1, 1});
                input_ids.emplace_back(bias_id);
            }

            // add conv node
            AddConvNode(scope_ptr, "output", input_ids,
                    {kernel_size, stride, padding});

            // add meta data
            scope->AddOutputName("output");
            return scope->ExportModel();
        }


        // create a session for all test
        FBOSession* InitSession(){
            FBOSession* session = new FBOSession;
            session->LoadGraph(BuildGraph());
            return session;
        }

        void SingleInference(){
            auto session = InitSession();

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
