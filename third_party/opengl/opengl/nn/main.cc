#include <stdio.h>
#include <memory>
#include <iostream>
#include <random>
#include <assert.h>
#include <cmath>
#include <cstring>
#include "opengl/core/init.h"
#include "opengl/core/types.h"
#include "opengl/utils/util.h"
#include <glog/logging.h>

#include "opengl/core/fbo_session.h"
#include "opengl/utils/env_time.h"
#include <opencv2/opencv.hpp>

// only tensor is visible in nn module
using opengl::Tensor;
using opengl::TensorList;
using opengl::FBOSession;

Tensor* PrepareInputs(std::string image_fname,
        const std::vector<int>& shape){
    auto raw_image = cv::imread(image_fname);
    cv::cvtColor(raw_image, raw_image, CV_BGR2RGB);
    cv::resize(raw_image, raw_image, cv::Size(shape[0], shape[1]));
    raw_image.convertTo(raw_image, CV_32FC3);
    const float mean_vals[3] = { 123.f, 117.f, 104.f};
    raw_image = raw_image - cv::Scalar(mean_vals[0], mean_vals[1], mean_vals[2]);

    ::opengl::IntList input_shape({1, shape[1], shape[0], 3});
    ::opengl::DataFormat input_dformat = ::dlxnet::TensorProto::NHWC;
    Tensor* input_tensor= Tensor::Zeros(Tensor::DT_FLOAT,
            input_shape, input_dformat);
    ::memcpy(input_tensor->host(), raw_image.data,
            input_tensor->num_elements()*sizeof(float));

    return input_tensor;
}

int main(int argc, char** argv){
    // Initialize Google's logging library.
    google::InitGoogleLogging(argv[0]);

    // glut_init(argc, argv);
    ::opengl::glfw_init();

    int maxtexsize;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE,&maxtexsize);
    LOG(INFO)<<"GL_MAX_TEXTURE_SIZE: "<<maxtexsize;

    // some params
    std::string model_path = "./demo.dlx";
    const int num_iters = 1;
    const float precision = 1e-6;

    // prepare inputs and outputs
    ::opengl::TensorList outputs_cpu;
    ::opengl::TensorNameList output_names({"cls_and_bbox"});

    // ::opengl::IntList input_shape({1, 320, 320, 3});
    // ::opengl::DataFormat input_dformat = ::dlxnet::TensorProto::NHWC;
    // Tensor* input_tensor= Tensor::Ones(Tensor::DT_FLOAT,
            // input_shape, input_dformat);
    std::string image_name = "/home/breakpoint/Documents/MNN/demo/test/demo.jpg";
    Tensor* input_tensor = PrepareInputs(image_name, {160, 160});
    ::opengl::StringList dformats({"ANY"});

    auto session = std::unique_ptr<FBOSession>(new FBOSession);
    session->LoadGraph(model_path);

    // warming up
    for(int i=0;i<3;++i){
        // init graph according to inputs
        // and then do computation for the graph
        session->Run({{"input", input_tensor}});

        // get cpu outputs from device
        session->GetOutputs(output_names, dformats, &outputs_cpu);

    }

    auto env_time = EnvTime::Default();
    auto start_time1 = env_time->NowMicros();
    std::string output_fn1 = "output.txt";
    std::string output_fn2 = "/home/breakpoint/Documents/Learning/cpp_best_practice/tools/converter/build/output2.txt";

    for(int i=0;i<num_iters;++i){
        // init graph according to inputs
        // do computation for the graph
        session->Run({{"input", input_tensor}});

        {
            // get cpu outputs from device
            session->GetOutputs(output_names, dformats, &outputs_cpu);
        }

        // print output
        LOG(INFO)<<outputs_cpu[0]->ShortDebugString();
        // dump output tensor
        // opengl::DumpTensor(outputs_cpu[0], output_fn1);
        // opengl::CompareTXT(output_fn1, output_fn2);
    }
    auto duration_time = env_time->NowMicros()-start_time1;
    // std::cout<<"Total Time: "<<duration_time*1e-3<<" ms\n";
    auto second_per_round = duration_time*1e-6/num_iters;
    // force to display
    std::cout<<"FPS: "<<1.0/second_per_round<<std::endl;

    LOG(INFO)<<"BiasAdd Success";

    return 0;
}
