#ifndef OPENGL_EXAMPLES_SSD_DETECTOR_H_
#define OPENGL_EXAMPLES_SSD_DETECTOR_H_
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include "common.h"

#include <chrono>


namespace opengl{
    struct DetectorOptions{
        string model_name;
        int input_width;
        int input_height;
        float nms_threshold;
        float score_threshold;
        int topk;
    };

    class Detector{
        public:
            virtual void Preprocess(const cv::Mat& image_in, cv::Mat& image_out);

            void NMS(std::vector<BoxInfo>& boxInfos,std::vector<BoxInfo>& boxInfos_left, float threshold);
            virtual void Detect(const cv::Mat& raw_image, std::vector<BoxInfo>& finalBoxInfos);

            virtual void Run(const cv::Mat& raw_image);

            virtual ~Detector();

            static std::unique_ptr<Detector> Create();
            static std::unique_ptr<Detector> Create(const DetectorOptions& options);


        private:
            explicit Detector(const DetectorOptions& options);
            // only uesd in postprocess
            void GetTopK(std::vector<BoxInfo>& input, int top_k);
            void GenerateBoxInfo(std::vector<BoxInfo>& boxInfos, float score_threshold);

            std::vector<float> mVariance;
            std::string mModelName;

            int topk_;
            float score_threshold_;
            float nms_threshold_;

            // num of total classes including bg
            int num_classes;

            // hw
            IntList input_sizes;
            IntList mOriginInputSize;

            // input and output tensors and its names
            std::vector<::opengl::Tensor*> mOutputTensors;
            std::vector<::opengl::Tensor*> mInputTensors;
            std::vector<std::string> mInputNames;
            std::vector<std::string> mOutputNames;

            std::vector<std::string> output_dformats_;


    };

}//namespace opengl

#endif
