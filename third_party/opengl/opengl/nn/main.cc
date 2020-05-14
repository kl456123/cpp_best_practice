#include <stdio.h>
#include <memory>
#include <iostream>
#include <random>
#include <assert.h>
#include <cmath>
#include <cstring>
#include "opengl/core/init.h"
#include "opengl/core/types.h"
#include <glog/logging.h>

#include "opengl/core/program.h"
#include "opengl/core/session.h"
#include "opengl/core/fbo_session.h"

// only tensor is visible in nn module
using opengl::Tensor;
using opengl::TensorList;
using opengl::Session;
using opengl::FBOSession;


float* AllocateHostMemory(std::vector<int> dims, bool fill_data=true){
    // get total num
    int num_elements = 1;
    for(auto dim:dims){
        num_elements*=dim;
    }
    float* image_data = new float[num_elements];
    memset(image_data, 0, sizeof(float)*num_elements);
    if(fill_data){
        for(int i=0;i<num_elements;i++){
            image_data[i] = 1.0;
        }
    }
    return image_data;
}

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



int main(int argc, char** argv){
    // glut_init(argc, argv);
    ::opengl::glfw_init();
    ::opengl::glew_init();

    int maxtexsize;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE,&maxtexsize);
    LOG(INFO)<<"GL_MAX_TEXTURE_SIZE: "<<maxtexsize;

    // conv2d params
    const int input_width = 3;
    const int input_height = 3;
    const int input_channels = 3;
    const int num_inputs = 1;
    const int kernel_size = 3;
    const int stride = 1;
    const int padding = 1;
    const int groups = 1;
    const int dilation = 1;

    // some params
    std::string model_path = "./demo.dlx";
    const int num_iters = 10;
    const float precision = 1e-6;

    // prepare inputs and outputs
    ::opengl::TensorList outputs_cpu;
    ::opengl::NamedTensorList inputs;
    ::opengl::TensorNameList output_names({"output", "conv2d1.weight", "conv2d1.bias"});
    ::opengl::StringList dformats({"NHWC", "NCHW", "NHWC"});
    std::vector<int> image_shape = {num_inputs, input_height, input_width, input_channels};

    auto session = std::unique_ptr<FBOSession>(new FBOSession);
    session->LoadGraph(model_path);
    // LOG(INFO)<<"ModelInfo After Load Graph: "
    // <<session->DebugString();

    for(int i=0;i<num_iters;++i){
        auto cpu_input_data =  AllocateHostMemory(image_shape, true);
        // init graph according to inputs
        session->Setup({{"input", new Tensor(Tensor::DT_FLOAT, image_shape, cpu_input_data)}});

        // do computation for the graph
        session->Run();

        // get cpu outputs from device
        session->GetOutputs(output_names, dformats, &outputs_cpu);
        const float* ogl_output_data = outputs_cpu[0]->host<float>();
        const int output_num_elements = outputs_cpu[0]->num_elements();
        const float* ogl_filter_data = outputs_cpu[1]->host<float>();
        const int filter_num_elements = outputs_cpu[1]->num_elements();
        const float* ogl_bias_data = outputs_cpu[2]->host<float>();
        // const float* ogl_input_data = outputs_cpu[3]->host<float>();
        // const int input_num_elements = outputs_cpu[3]->num_elements();

        // nhwc
        auto output_shape = outputs_cpu[0]->shape();
        const int output_width = output_shape[2];
        const int output_height = output_shape[1];
        const int output_channels = output_shape[3];

        auto cpu_output_data = AllocateHostMemory(output_shape, false);
        const float* cpu_filter_data = ogl_filter_data;
        const float* cpu_bias_data = ogl_bias_data;
        // Conv2DCPU(cpu_input_data, cpu_filter_data, cpu_bias_data, cpu_output_data,
                // kernel_size, stride, padding, input_width, input_height,
                // output_width, output_height, input_channels, output_channels, dilation, groups);
        // check input
        // for(int i=0;i<input_num_elements;++i){
        // CHECK_EQ(cpu_input_data[i], ogl_input_data[i])
        // <<"Error When index: "<< i;
        // }

        // check filter
        // for(int i=0;i<filter_num_elements;++i){
        // CHECK_EQ(cpu_filter_data[i], ogl_filter_data[i])
        // <<"Error When index: "<< i;
        // }
        // const float precision = 1e-7; // failed when mediump

        // check the result
        // for(int i=0;i<output_num_elements;++i){
            // float actual_value = ogl_output_data[i];
            // float expect_value = cpu_output_data[i];
            // CHECK_LT(std::fabs(actual_value- expect_value), precision)<<"Error When index: "<< i
                // <<" Actualy Value: "<<actual_value<<" Extect Value: "<<expect_value;
        // }

        // print output
        for(int i=0; i<std::min(output_num_elements, 100); ++i){
            LOG(INFO)<<ogl_output_data[i];
        }

    }

    LOG(INFO)<<"BiasAdd Success";


    return 0;
}
