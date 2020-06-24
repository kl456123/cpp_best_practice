#include "opengl/utils/logging.h"
#include <string.h>
#include "opengl/core/fbo_session.h"
#include "opengl/core/tensor.h"
#include "opengl/core/init.h"
#include "opengl/core/functor.h"
#include "opengl/examples/ssd/detector.h"

#include <opencv2/opencv.hpp>

using opengl::FBOSession;
using opengl::Tensor;
using opengl::Detector;

namespace functor = opengl::functor;

int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);
    ::opengl::glfw_init();

    // config detector
    std::string model_path = "./demo.dlx";
    auto detector = Detector::Create(model_path, {"input"},
            {"cls_and_bbox", "anchors"});

    // prepare inputs
    std::string image_fname = "../opengl/examples/ssd/000000145679.jpg";
    auto raw_image = cv::imread(image_fname);
    // graylize first
    cv::cvtColor(raw_image, raw_image, CV_BGR2GRAY);
    cv::cvtColor(raw_image, raw_image, CV_GRAY2BGR);

    while(true){
        auto t1 = std::chrono::system_clock::now();
        // detect
        std::vector<BoxInfo> finalBoxInfos;

        detector->Detect(raw_image, finalBoxInfos);
        auto t2 = std::chrono::system_clock::now();
        float dur = (float)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000;
        std::cout << "duration time:" << dur << "ms" << std::endl;

        drawBoxes(finalBoxInfos, raw_image);
        cv::namedWindow("dlxnet", CV_WINDOW_NORMAL);
        cv::imshow("dlxnet", raw_image);
        cv::waitKey(1);
    }
    return 0;
}
