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

// only tensor is visible in nn module
using opengl::Tensor;
using opengl::TensorList;
using opengl::Session;


float* AllocateHostMemory(std::vector<int> dims, bool fill_data=true){
    // get total num
    int num_elements = 1;
    for(auto dim:dims){
        num_elements*=dim;
    }
    float* image_data = new float[num_elements];
    if(fill_data){
        for(int i=0;i<num_elements;i++){
            image_data[i] = random()%256/256.0;
        }
    }
    return image_data;
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
    int num_inputs = 2;
    int num_outputs= 1;
    std::vector<int> tensor_shape = {225, 225, 4};
    auto input_data =  AllocateHostMemory(tensor_shape, true);
    auto output_data = AllocateHostMemory(tensor_shape, false);
    for(int i=0;i<num_inputs;++i){
        inputs_cpu.emplace_back(new Tensor(input_data, Tensor::DT_FLOAT, tensor_shape));
    }

    for(int i=0;i<num_outputs;++i){
        outputs_cpu.emplace_back(new Tensor(output_data, Tensor::DT_FLOAT, tensor_shape));
    }


    auto session = std::unique_ptr<Session>(new Session);

    session->LoadGraph({"Add"});

    // init graph according to inputs
    session->Setup(inputs_cpu);

    // do computation for the graph
    session->Run();

    // get cpu outputs from device
    session->GetOutputs(outputs_cpu);

    // check the result
    // CheckTensorSame(outputs_cpu);
    for(int i=0;i<10;++i){
        float actual_value = outputs_cpu[0]->host<float>()[i];
        float expect_value = 2* input_data[i];
        CHECK_EQ(actual_value, expect_value)
            <<"Expect Value: "<<expect_value
            <<"Actual Value: "<<actual_value;
    }

    LOG(INFO)<<"BiasAdd Success";


    return 0;
}
