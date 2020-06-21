#include "opengl/utils/logging.h"
#include <string.h>
#include "opengl/core/fbo_session.h"
#include "opengl/core/tensor.h"
#include "opengl/core/init.h"
#include "opengl/core/functor.h"

#include <opencv2/opencv.hpp>

using opengl::FBOSession;
using opengl::Tensor;
namespace functor = opengl::functor;

int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);
    ::opengl::glfw_init();

    // load model
    std::string model_path = "./demo.dlx";
    auto session = std::unique_ptr<FBOSession>(new FBOSession);
    session->LoadGraph(model_path);

    // prepare input image
    ::opengl::IntList input_shape({1, 320, 320, 3});
    ::opengl::DataFormat input_dformat = ::dlxnet::TensorProto::NHWC;
    Tensor* image_ptr = Tensor::Ones(Tensor::DT_FLOAT,
            input_shape, input_dformat);

    ::opengl::TensorList outputs_cpu;

    // Run Session with input
    // do computation for the graph
    session->Run({{"input", image_ptr}});

    // postprocess for detected box
    Tensor* anchors = session->FindTensorByName("anchors");// not owned
    // get encoded prediction from session directly
    Tensor* prediction = session->FindTensorByName("cls_and_bbox");// not owned
    const int num_samples = prediction->shape()[1];
    const int topk = 100;
    const float nms_threshold = 0.45;

    auto boxes_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT,
                {1, num_samples, 4}, Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY4));
    auto final_boxes_gpu_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT,
                {1, topk, 4}, Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY4));
    Tensor* boxes = boxes_ptr.get();
    Tensor* final_boxes_gpu = final_boxes_gpu_ptr.get();
    functor::SSDBBoxDecoder()(session->context(), prediction, anchors, boxes,
            {0.1, 0.1, 0.2, 0.2}/*variances*/);
    functor::NMS()(session->context(), boxes, final_boxes_gpu, nms_threshold);

    // copy to gpu
    auto final_boxes_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT,
                final_boxes_gpu->shape(), dlxnet::TensorProto::ANY));
    auto final_boxes = final_boxes_ptr.get();
    session->context()->CopyDeviceTensorToCPU(final_boxes_gpu, final_boxes);
    LOG(INFO)<<final_boxes->ShortDebugString();

    // draw boxes in the image and save it
    return 0;
}
