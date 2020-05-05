#include <stdio.h>
#include <memory>
#include <iostream>
#include <random>
#include <assert.h>
#include <cstring>
#include "opengl/core/init.h"
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
        int num_channels){
    // 1D representation of hwc for all args
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
                    for(int c=0;c<num_channels;++c){
                        float a = input_data[input_index*num_channels+c];
                        float b = filter_data[filter_index*num_channels+c];
                        sum+=a*b;
                    }
                }
            }
            output_data[output_index*num_channels] = sum;
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
    TensorList inputs_cpu, outputs_cpu;
    const int input_width = 224;
    const int input_height = 224;
    const int output_width = 224;
    const int output_height = 224;

    std::vector<int> image_shape = {input_height, input_width, 4};
    std::vector<int> filter_shape = {3, 3, 4};
    std::vector<int> output_shape = {output_height,output_width,4};
    auto input_data =  AllocateHostMemory(image_shape, true);
    auto filter_data =  AllocateHostMemory(image_shape, true);
    auto output_data = AllocateHostMemory(output_shape, false);
    auto output_data2 = AllocateHostMemory(output_shape, false);
    inputs_cpu.emplace_back(new Tensor(input_data, Tensor::DT_FLOAT, image_shape));
    inputs_cpu.emplace_back(new Tensor(filter_data, Tensor::DT_FLOAT, filter_shape));

    outputs_cpu.emplace_back(new Tensor(output_data, Tensor::DT_FLOAT, output_shape));

    auto session = std::unique_ptr<FBOSession>(new FBOSession);

    session->LoadGraph({"Conv2d"});

    // init graph according to inputs
    session->Setup(inputs_cpu);

    // do computation for the graph
    session->Run();

    // get cpu outputs from device
    session->GetOutputs(outputs_cpu);

    Conv2DCPU(input_data, filter_data, output_data2, 3,1,1,input_width,input_height,
            output_width,output_height, 4);

    // check the result
    // CheckTensorSame(outputs_cpu);
    for(int i=0;i<outputs_cpu[0]->num_elements();++i){
        float actual_value = outputs_cpu[0]->host<float>()[i];
        float expect_value = output_data2[i];
        CHECK_EQ(actual_value, expect_value)<<"Error When index: "<< i;
    }

    LOG(INFO)<<"BiasAdd Success";


    return 0;
}
