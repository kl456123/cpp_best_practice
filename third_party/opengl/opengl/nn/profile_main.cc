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
#include "opengl/core/lib/monitor/collection_registry.h"
#include "opengl/nn/profiler/profiler.h"
#include "opengl/core/step_stats.pb.h"
#include "opengl/nn/profiler/profiler_session.h"

// only tensor is visible in nn module
using opengl::Tensor;
using opengl::TensorList;
using opengl::Session;
using opengl::FBOSession;
using opengl::monitoring::CollectionRegistry;
using opengl::monitoring::CollectedMetrics;
using opengl::Profiler;
using opengl::ProfilerSession;

namespace{
    void ProfileProgram(){
        // collect stats of graph runtime
        auto* collection_registry = CollectionRegistry::Default();
        const std::unique_ptr<CollectedMetrics> collected_metrics =
            collection_registry->CollectMetrics({});
        CHECK_GT(collected_metrics->metric_descriptor_map.size(), 0);
        CHECK_GT(collected_metrics->point_set_map.size(), 0);

        // execution time
        auto loop_times = collected_metrics->point_set_map[
            "/tensorflow/core/graph_runs"]->points[0]->int64_value;
        auto graph_run_time_usecs = collected_metrics->point_set_map[
            "/tensorflow/core/graph_run_time_usecs"]->points[0]->int64_value;
        std::cout<<"ExecTimePerRound: "<<graph_run_time_usecs / loop_times/1e3<<" ms"<<std::endl;

        // build time
        //
        auto build_times = collected_metrics->point_set_map[
            "/tensorflow/core/graph_build_calls"]->points[0]->int64_value;
        auto graph_build_time_usecs = collected_metrics->point_set_map[
            "/tensorflow/core/graph_build_time_usecs"]->points[0]->int64_value;
        std::cout<<"BuildTimePerRound: "<<graph_build_time_usecs/build_times/1e3<<" ms"<<std::endl;
    }
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
    const int num_iters = 10;
    const float precision = 1e-6;

    // prepare inputs and outputs
    ::opengl::TensorList outputs_cpu;
    ::opengl::TensorNameList output_names({"cls_and_bbox"});
    ::opengl::IntList input_shape({1, 320, 320, 3});
    ::opengl::DataFormat input_dformat = ::dlxnet::TensorProto::NHWC;
    Tensor* input_tensor= Tensor::Ones(Tensor::DT_FLOAT,
            input_shape, input_dformat);
    auto profiler = std::unique_ptr<Profiler>(new Profiler);
    // ::opengl::StringList dformats({"NHWC"});
    // ::opengl::DataFormat input_dformat = ::dlxnet::TensorProto::ANY;
    ::opengl::StringList dformats({"ANY"});

    auto session = std::unique_ptr<FBOSession>(new FBOSession);
    session->LoadGraph(model_path);
    // LOG(INFO)<<"ModelInfo After Load Graph: "
    // <<session->DebugString();

    // warming up
    for(int i=0;i<3;++i){
        // init graph according to inputs
        // and then do computation for the graph
        session->Run({{"input", Tensor::Ones(Tensor::DT_FLOAT,
                    input_shape, input_dformat)}});

        // get cpu outputs from device
        session->GetOutputs(output_names, dformats, &outputs_cpu);

    }

    auto profiler_session = ProfilerSession::Create();
    for(int i=0;i<num_iters;++i){
        // init graph according to inputs
        // do computation for the graph
        {
            session->Run({{"input", input_tensor}});
        }

        {
            // get cpu outputs from device
            session->GetOutputs(output_names, dformats, &outputs_cpu);
        }
    }

    ::opengl::StepStats step_stats;
    profiler_session->CollectData(&step_stats);
    profiler->CollectData(&step_stats);

    std::cout<<"----------------------profiler start------------------"<<std::endl;
    profiler->PrintProfiling(num_iters);
    std::cout<<"----------------------profiler end------------------"<<std::endl;

    return 0;
}
