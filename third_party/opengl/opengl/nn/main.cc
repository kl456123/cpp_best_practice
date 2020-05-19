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


int main(int argc, char** argv){
    // Initialize Google's logging library.
    google::InitGoogleLogging(argv[0]);

    // glut_init(argc, argv);
    ::opengl::glfw_init();
    ::opengl::glew_init();

    int maxtexsize;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE,&maxtexsize);
    LOG(INFO)<<"GL_MAX_TEXTURE_SIZE: "<<maxtexsize;

    // conv2d params
    const int input_width = 224;
    const int input_height = 224;
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
    ::opengl::TensorNameList output_names({"output"});
    ::opengl::StringList dformats({"NHWC"});
    std::vector<int> image_shape = {num_inputs, input_height, input_width, input_channels};

    inputs.resize(1);
    auto session = std::unique_ptr<FBOSession>(new FBOSession);
    session->LoadGraph(model_path);
    // LOG(INFO)<<"ModelInfo After Load Graph: "
    // <<session->DebugString();

    for(int i=0;i<num_iters;++i){
        // auto cpu_input_data =  AllocateHostMemory(image_shape, true);
        // init graph according to inputs
        inputs[0].first = "input";
        inputs[0].second = Tensor::Random(Tensor::DT_FLOAT, image_shape);
        session->Setup(inputs);

        // do computation for the graph
        session->Run();

        // get cpu outputs from device
        session->GetOutputs(output_names, dformats, &outputs_cpu);
        const float* ogl_output_data = outputs_cpu[0]->host<float>();
        const int output_num_elements = outputs_cpu[0]->num_elements();
        // const float* ogl_filter_data = outputs_cpu[1]->host<float>();
        // const int filter_num_elements = outputs_cpu[1]->num_elements();
        // const float* ogl_bias_data = outputs_cpu[2]->host<float>();
        // const float* ogl_input_data = outputs_cpu[3]->host<float>();
        // const int input_num_elements = outputs_cpu[3]->num_elements();

        // nhwc
        auto output_shape = outputs_cpu[0]->shape();
        const int output_width = output_shape[2];
        const int output_height = output_shape[1];
        const int output_channels = output_shape[3];

        // print output
        for(int i=0; i<std::min(output_num_elements, 100); ++i){
            LOG(INFO)<<ogl_output_data[i];
        }

    }

    LOG(INFO)<<"BiasAdd Success";


    return 0;
}
