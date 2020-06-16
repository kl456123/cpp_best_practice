#include "opengl/utils/logging.h"
#include <string.h>
#include "opengl/core/fbo_session.h"
#include "opengl/core/tensor.h"
#include "opengl/core/init.h"

#include <opencv2/opencv.hpp>

using opengl::FBOSession;
using opengl::Tensor;

int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);
    ::opengl::glfw_init();

    // load model
    std::string model_path = "./demo.dlx";
    auto session = std::unique_ptr<FBOSession>(new FBOSession);
    session->LoadGraph(model_path);

    Tensor* image_ptr;

    ::opengl::TensorList outputs_cpu;

    // Run Session with input
    // do computation for the graph
    session->Run({{"input", image_ptr}});

    // get cpu outputs from device
    session->GetOutputs({"output"}, {"ANY"}, &outputs_cpu);

    // postprocess for detected box
    //
    //
    // draw boxes in the image and save it
    return 0;
}
