#include "opengl/examples/ssd/detector.h"
#include "opengl/core/fbo_session.h"


namespace opengl{
    /*static*/ std::unique_ptr<Detector> Detector::Create(){
        // use default options here
        DetectorOptions options;
        options.topk = 100;
        return Create(options);
    }

    /*static*/ std::unique_ptr<Detector> Detector::Create(const DetectorOptions& options){
        return std::unique_ptr<Detector>(new Detector(options));
    }

    Detector::Detector(const DetectorOptions& options){
        mVariance = {0.1, 0.1, 0.2, 0.2};

        score_threshold_ = options.score_threshold;
        nms_threshold_= options.nms_threshold;
        topk_ = options.topk;

        input_sizes_.push_back(options.input_height);
        input_sizes_.push_back(options.input_width);

        session_.reset(new FBOSession);
        session_->LoadGraph(options.model_name);

        // allocate memory for input tensor
        ::opengl::IntList input_shape({1, 320, 320, 3});
        ::opengl::DataFormat input_dformat = ::dlxnet::TensorProto::NHWC;
        ::opengl::Tensor* image_ptr = Tensor::Ones(Tensor::DT_FLOAT,
                input_shape, input_dformat);

        input_tensors_.emplace_back(image_ptr);
        output_tensors_.clear();
        output_names_ = {"cls_and_bbox", "anchors"};
        input_names_ = {"input"};
        output_dformats_ = {"ANY", "ANY"};
    }



    void Detector::Preprocess(const cv::Mat& raw_image, cv::Mat& image){
        cv::cvtColor(raw_image,image, cv::COLOR_BGR2RGB);

        origin_input_sizes_ = {raw_image.rows, raw_image.cols};
        // order? hw or wh
        cv::resize(image, image, cv::Size(input_sizes_[1], input_sizes_[0]));

        image.convertTo(image, CV_32FC3);
        const float mean_vals[3] = { 123.f, 117.f, 104.f};
        image = image - cv::Scalar(mean_vals[0], mean_vals[1], mean_vals[2]);
    }




    void Detector::GetTopK(std::vector<BoxInfo>& input, int top_k)
    {
        std::sort(input.begin(), input.end(),
                [](const BoxInfo& a, const BoxInfo& b)
                {
                return a.score > b.score;
                });

        if(top_k<input.size()){
            input.erase(input.begin()+top_k, input.end());
        }
    }

    void Detector::NMS(std::vector<BoxInfo>& tmp_faces, std::vector<BoxInfo>& faces, float nms_threshold){
        int N = tmp_faces.size();
        std::vector<int> labels(N, -1);
        for(int i = 0; i < N-1; ++i)
        {
            for (int j = i+1; j < N; ++j)
            {
                cv::Rect pre_box = tmp_faces[i].box;
                cv::Rect cur_box = tmp_faces[j].box;
                float iou_ = iou(pre_box, cur_box);
                if (iou_ > nms_threshold) {
                    labels[j] = 0;
                }
            }
        }

        for (int i = 0; i < N; ++i)
        {
            if (labels[i] == -1)
                faces.push_back(tmp_faces[i]);
        }
    }

    void Detector::GenerateBoxInfo(std::vector<BoxInfo>& boxInfos, float score_threshold){
        auto tensors_host = output_tensors_;

        // auto scores_dataPtr  = tensors_host[0]->host<float>();
        // auto boxes_dataPtr   = tensors_host[1]->host<float>();
        auto scores_and_boxes_dataPtr = tensors_host[0]->host<float>();
        auto anchors_dataPtr = tensors_host[1]->host<float>();
        int num_boxes = tensors_host[0]->channel();
        int raw_image_width = origin_input_sizes_[1];
        int raw_image_height = origin_input_sizes_[0];
        num_classes_ = tensors_host[0]->shape()[3]-4;
        int num_cols = num_classes_+4;

        for(int i = 0; i < num_boxes; ++i)
        {
            // location decoding
            float ycenter =     scores_and_boxes_dataPtr[i*num_cols + +num_classes_+1] * mVariance[1]  * anchors_dataPtr[i*4 + 3] + anchors_dataPtr[i*4 + 1];
            float xcenter =     scores_and_boxes_dataPtr[i*num_cols + num_classes_+0] * mVariance[0]  * anchors_dataPtr[i*4 + 2] + anchors_dataPtr[i*4 + 0];
            float h       = exp(scores_and_boxes_dataPtr[i*num_cols + num_classes_+3] * mVariance[3]) * anchors_dataPtr[i*4 + 3];
            float w       = exp(scores_and_boxes_dataPtr[i*num_cols + num_classes_+2] * mVariance[2]) * anchors_dataPtr[i*4 + 2];

            float ymin    = ( ycenter - h * 0.5 ) * raw_image_height;
            float xmin    = ( xcenter - w * 0.5 ) * raw_image_width;
            float ymax    = ( ycenter + h * 0.5 ) * raw_image_height;
            float xmax    = ( xcenter + w * 0.5 ) * raw_image_width;

            // probability decoding, softmax
            float total_sum = exp(scores_and_boxes_dataPtr[i*num_cols + 0]);
            // init
            int max_id = 0;
            float max_prob=0;

            for(int j=1;j<num_classes_;j++){
                float logit = exp(scores_and_boxes_dataPtr[i*num_cols + j]);
                total_sum  += logit;
                if(max_prob<logit){
                    max_prob = logit;
                    max_id = j;
                }
            }

            max_prob /= total_sum;


            if (max_prob > score_threshold) {
                BoxInfo tmp_face;
                tmp_face.box.x = xmin;
                tmp_face.box.y = ymin;
                tmp_face.box.width  = xmax - xmin;
                tmp_face.box.height = ymax - ymin;

                // center
                tmp_face.cx = (xmin+xmax)/2.0;
                tmp_face.cy = (ymin+ymax)/2.0;

                tmp_face.height = tmp_face.box.height;
                tmp_face.width = tmp_face.box.width;

                tmp_face.score = max_prob;
                tmp_face.class_name = static_cast<CLASS_NAME>(max_id);
                boxInfos.push_back(tmp_face);
            }
        }
    }

    void Detector::Run(){
        // Run Session with input
        // do computation for the graph
        session_->Run({{"input", input_tensors_[0]}});
        session_->GetOutputs(output_names_, output_dformats_, &output_tensors_);
    }


    void Detector::Detect(const cv::Mat& raw_image, std::vector<BoxInfo>& finalBoxInfos){
        // preprocess
        cv::Mat image;
        std::cout<<"Preprocessing "<<std::endl;
        Preprocess(raw_image, image);


        std::cout<<"Running "<<std::endl;
        Run();

        std::cout<<"Postprocessing "<<std::endl;
        // postprocess
        std::vector<BoxInfo> boxInfos;
        GenerateBoxInfo(boxInfos, score_threshold_);
        // top k
        GetTopK(boxInfos, topk_);
        // nms
        NMS(boxInfos, finalBoxInfos, nms_threshold_);

        // handle corner case
    }


    Detector::~Detector(){
        // clear input and output tensors
        for(auto tensor:output_tensors_){
            if(tensor){
                delete tensor;
            }
        }

        for(auto tensor:input_tensors_){
            if(tensor){
                delete tensor;
            }
        }
    }
} // namespace opengl
