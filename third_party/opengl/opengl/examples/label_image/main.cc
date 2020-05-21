/* Image Classification Task, Input a single image,  Output confidences for multiple
 * predefined classes(1000 classes in ImageNet).
 *
 */
#include <glog/logging.h>
#include "opengl/core/fbo_session.h"
#include "opengl/core/tensor.h"

// #include <opencv2/opencv.hpp>

using opengl::FBOSession;
using opengl::Tensor;

bool ReadImage(const char* image_name, Tensor* image){
    // read image and store them in image ptr in nhwc dformat
    return true;
}


int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);

    // load model
    std::string model_path = "./demo.dlx";
    auto session = std::unique_ptr<FBOSession>(new FBOSession);
    session->LoadGraph(model_path);

    // prepare input
    // load image to input tensor
    const char* image_name= "demo.jpg";
    Tensor* image_ptr;
    bool success = ReadImage(image_name, image_ptr);
    if(!success){
        LOG(FATAL)<<"Read Image Failed";
        return -1;
    }

    ::opengl::NamedTensorList inputs;
    ::opengl::TensorList outputs_cpu;
    inputs.resize(1);
    inputs[0].first = "input";
    inputs[0].second = image_ptr;

    // Run Session with input
    session->Setup(inputs);

    // do computation for the graph
    session->Run();

    // get cpu outputs from device
    session->GetOutputs({"output"}, {"NHWC"}, &outputs_cpu);

    // print output
    LOG(INFO)<<outputs_cpu[0]->ShortDebugString();
    return 0;
}
