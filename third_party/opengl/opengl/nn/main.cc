#include <stdio.h>
#include <memory>
#include <iostream>
#include <random>
#include <assert.h>
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
            image_data[i] = random()%256/256.0;
        }
    }
    return image_data;
}

void Conv2DCPU(const float* input_data,
        const float* filter_data,
        float* output_data,
        int kernel_size,
        int stride,
        int padding,
        int input_width,
        int input_height,
        int output_width,
        int output_height,
        int input_channels,
        int output_channels){
    // 1D representation of hwc for all args
    for(int oc=0;oc<output_channels;++oc){
        const int filter_base = oc*kernel_size*kernel_size*input_channels;
        for(int i=0;i<output_height;++i){
            for(int j=0;j<output_width;++j){
                int output_index = i*output_width+j;
                float sum = 0;
                for(int r=0;r<kernel_size;++r){
                    for(int s=0;s<kernel_size;++s){
                        int input_index_x = j*stride-padding+s;
                        int input_index_y = i*stride-padding+r;
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
                            float b = filter_data[filter_base+filter_index*input_channels+c];
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

    // prepare inputs and outputs
    ::opengl::TensorList outputs_cpu;
    ::opengl::NamedTensorList inputs;
    ::opengl::TensorNameList output_names({"output"});
    const int input_width = 224;
    const int input_height = 224;
    const int output_width = 224;
    const int output_height = 224;
    const int input_channels = 3;
    const int output_channels = 1;

    std::vector<int> image_shape = {input_height, input_width, input_channels};
    std::vector<int> filter_shape = {output_channels, 3, 3, input_channels};
    std::vector<int> output_shape = {output_height,output_width,output_channels};
    auto input_data =  AllocateHostMemory(image_shape, true);
    auto filter_data =  AllocateHostMemory(filter_shape, true);
    auto output_data = AllocateHostMemory(output_shape, false);
    // inputs.emplace_back();

    auto session = std::unique_ptr<FBOSession>(new FBOSession);
    std::string model_path = "./demo.dlx";

    session->LoadGraph(model_path);
    LOG(INFO)<<"ModelInfo After Load Graph: "
        <<session->DebugString();

    // init graph according to inputs
    session->Setup({{"input", new Tensor(Tensor::DT_FLOAT, image_shape, input_data)}});

    // do computation for the graph
    session->Run();

    // get cpu outputs from device
    session->GetOutputs(output_names, &outputs_cpu);

    Conv2DCPU(input_data, filter_data, output_data, 3,1,1, input_width, input_height,
            output_width,output_height, input_channels, output_channels);

    // check the result
    // CheckTensorSame(outputs_cpu);
    for(int i=0;i<outputs_cpu[0]->num_elements();++i){
        float actual_value = outputs_cpu[0]->host<float>()[i];
        float expect_value = input_data[i];
        CHECK_EQ(actual_value, expect_value)<<"Error When index: "<< i;
    }

    LOG(INFO)<<"BiasAdd Success";


    return 0;
}
