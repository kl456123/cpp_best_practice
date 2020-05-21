#include "opengl/core/tensor_format.h"

namespace opengl{
    bool FormatFromString(const std::string& format_str, TensorFormat* format) {
        if (format_str == "NHWC") {
            *format = TensorFormat::NHWC;
            return true;
        }
        if (format_str == "NCHW") {
            *format = TensorFormat::NCHW;
            return true;
        }
        if (format_str == "NCHW4") {
            *format = TensorFormat::NCHW4;
            return true;
        }
        if (format_str == "NHWC4") {
            *format = TensorFormat::NHWC4;
            return true;
        }
        return false;
    }

    bool FilterFormatFromString(const std::string& format_str,
            FilterTensorFormat* format) {
        if (format_str == "OIHW") {
            *format = FilterTensorFormat::OIHW;
            return true;
        }
        if (format_str == "HWI4O4") {
            *format = FilterTensorFormat::HWI4O4;
            return true;
        }
        if (format_str == "HWIO") {
            *format = FilterTensorFormat::HWIO;
            return true;
        }
        return false;
    }

    std::string ToString(TensorFormat format) {
        switch (format) {
            case TensorFormat::NHWC:
                return "NHWC";
            case TensorFormat::NCHW:
                return "NCHW";
            case TensorFormat::NCHW4:
                return "NCHW4";
            case TensorFormat::NHWC4:
                return "NHWC4";
            default:
                LOG(FATAL) << "Invalid Format: " << static_cast<int32_t>(format);
                return "INVALID_FORMAT";
        }
    }

    std::string ToString(FilterTensorFormat format){
        switch (format) {
            case FilterTensorFormat::HWIO:
                return "HWIO";
            case FilterTensorFormat::HWI4O4:
                return "HWI4O4";
            case FilterTensorFormat::OIHW:
                return "OIHW";
            default:
                LOG(FATAL) << "Invalid Filter Format: " << static_cast<int32_t>(format);
                return "INVALID_FORMAT";
        }
    }
}//namespace opengl
