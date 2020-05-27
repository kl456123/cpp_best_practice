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
#include "opengl/utils/env_time.h"

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

    int maxtexsize;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE,&maxtexsize);
    LOG(INFO)<<"GL_MAX_TEXTURE_SIZE: "<<maxtexsize;

    // conv2d params
    const int input_width = 224;
    const int input_height = 224;
    const int input_channels = 3;
    const int num_inputs = 1;

    // some params
    std::string model_path = "./demo.dlx";
    const int num_iters = 200;
    const float precision = 1e-6;

    // prepare inputs and outputs
    ::opengl::TensorList outputs_cpu;
    ::opengl::TensorNameList output_names({"output"});
    ::opengl::StringList dformats({"NHWC"});

    auto session = std::unique_ptr<FBOSession>(new FBOSession);
    session->LoadGraph(model_path);
    // LOG(INFO)<<"ModelInfo After Load Graph: "
    // <<session->DebugString();

    // warming up
    for(int i=0;i<3;++i){
        // init graph according to inputs
        session->Setup({{"input", Tensor::Ones(Tensor::DT_FLOAT,
                    {num_inputs, input_height, input_width, input_channels})}});
        // do computation for the graph
        session->Run();

        // get cpu outputs from device
        session->GetOutputs(output_names, dformats, &outputs_cpu);

        // print output
        // LOG(INFO)<<outputs_cpu[0]->ShortDebugString();

    }
    auto env_time = EnvTime::Default();
    auto start_time = env_time->NowMicros();
    for(int i=0;i<num_iters;++i){
        // init graph according to inputs
        session->Setup({{"input", Tensor::Ones(Tensor::DT_FLOAT,
                    {num_inputs, input_height, input_width, input_channels})}});
        // do computation for the graph
        session->Run();

        // get cpu outputs from device
        session->GetOutputs(output_names, dformats, &outputs_cpu);

        // print output
        // LOG(INFO)<<outputs_cpu[0]->ShortDebugString();

    }
    auto duration_time = env_time->NowMicros()-start_time;
    auto second_per_round = duration_time*1e-6/num_iters;
    // force to display
    std::cout<<"FPS: "<<second_per_round<<std::endl;

    LOG(INFO)<<"BiasAdd Success";


    return 0;
}
