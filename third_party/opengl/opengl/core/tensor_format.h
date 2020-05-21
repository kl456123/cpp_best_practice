#ifndef OPENGL_CORE_TENSOR_FORMAT_H_
#define OPENGL_CORE_TENSOR_FORMAT_H_
/* Tensor Format give means for each dim in tensor.
 * for a tensor of shape like (1,3,4,4), if use nhwc to interpret it,
 * 4 means 4 channels, 3 means height and so on.
 *
 */

#include <string>
#include <glog/logging.h>
#include "opengl/core/types.h"

namespace opengl{
    // tensor format in each node in output and input
    enum class TensorFormat{
        // N means batch, H means height, W means width, C means channel

        // list its order as batch dim , height dim , width dim, channel dim
        // it is most commonly used by tensorflow. Considering data contiugous in channel dim,
        // we use it as dformat in node of input and output in dlxnet framework
        NHWC,

        // Commonly used by pytorch and caffe.
        NCHW,

        // other than that pack 4 adjacent input pixel in channels, the
        // remain things is the same with nhwc dformat
        NHWC4,

        // pack 4 adjacent input pixel in width
        NCHW4
    };

    // filter tensor format, it is different fromt tensor format,
    // only used in convolution
    // note that O refers to out_channels, I refers to in_channels, H refers to height, W refers to width
    enum class FilterTensorFormat{
        // used by pytorch and caffe, can be considered as "nchw" in filter format
        OIHW,

        // continugous in out channel to calculate multiple out pixel at the same time
        HWIO,

        // used in dlxnet framework, to pack 4 in channels and out channels
        // H, W, O/4, I/4, I4, O4
        HWI4O4
    };

    // some funcs to use specified format to get the axis of the meaning
    // dim like height, channels and so on

    // Parse tensor format from the given string.
    // Return true if the parsing succeeds, and false if it fails.
    bool FormatFromString(const std::string& format_str, TensorFormat* format);

    // Parse tensor format from the given string.
    // Return true if the parsing succeeds, and false if it fails.
    bool FilterFormatFromString(const std::string& format_str,
            FilterTensorFormat* format);

    // Convert a tensor format into string.
    std::string ToString(TensorFormat format);

    // Convert a filter tensor format into string.
    std::string ToString(FilterTensorFormat format);

    // Returns the index of the batch dimension.
    // note that num_dims can large than 4
    inline int GetTensorBatchDimIndex(int num_dims, TensorFormat format) {
        switch (format) {
            case TensorFormat::NHWC:
            case TensorFormat::NCHW:
            case TensorFormat::NHWC4:
            case TensorFormat::NCHW4:
                return 0;
            default:
                LOG(FATAL) << "Unknown format " << static_cast<int32_t>(format);
                return -1;  // Avoid compiler warning about missing return value
        }
    }

    // Returns the index of the channel dimension
    // when format is packed, return the outer one, holding C/4
    inline int GetTensorChannelDimIndex(int num_dims, TensorFormat format){
        switch (format) {
            case TensorFormat::NHWC:
                return num_dims-1;
            case TensorFormat::NCHW:
                return num_dims-3;
            case TensorFormat::NHWC4:
                return num_dims-2;
            case TensorFormat::NCHW4:
                return num_dims-4;
            default:
                LOG(FATAL) << "Unknown format " << static_cast<int32_t>(format);
                return -1;  // Avoid compiler warning about missing return value
        }
    }

    // returns height dim index
    inline int GetTensorHeightDimIndex(int num_dims, TensorFormat format){
        switch (format) {
            case TensorFormat::NHWC:
                return num_dims-3;
            case TensorFormat::NCHW:
                return num_dims-2;
            case TensorFormat::NHWC4:
                return num_dims-4;
            case TensorFormat::NCHW4:
                return num_dims-3;
            default:
                LOG(FATAL) << "Unknown format " << static_cast<int32_t>(format);
                return -1;  // Avoid compiler warning about missing return value
        }
    }

    // returns width dim index
    inline int GetTensorWidthDimIndex(int num_dims, TensorFormat format){
        switch (format) {
            case TensorFormat::NHWC:
                return num_dims-2;
            case TensorFormat::NCHW:
                return num_dims-1;
            case TensorFormat::NHWC4:
                return num_dims-3;
            case TensorFormat::NCHW4:
                return num_dims-2;
            default:
                LOG(FATAL) << "Unknown format " << static_cast<int32_t>(format);
                return -1;  // Avoid compiler warning about missing return value
        }
    }

    inline int GetFilterTensorWidthDimIndex(int num_dims, FilterTensorFormat format){
        switch (format) {
            case FilterTensorFormat::OIHW:
                return num_dims-1;
            case FilterTensorFormat::HWIO:
                return num_dims-3;
            case FilterTensorFormat::HWI4O4:
                return num_dims-5;
            default:
                LOG(FATAL) << "Unknown format " << static_cast<int32_t>(format);
                return -1;  // Avoid compiler warning about missing return value
        }
    }

    inline int GetFilterTensorHeightDimIndex(int num_dims, FilterTensorFormat format){
        switch (format) {
            case FilterTensorFormat::OIHW:
                return num_dims-2;
            case FilterTensorFormat::HWIO:
                return num_dims-4;
            case FilterTensorFormat::HWI4O4:
                return num_dims-6;
            default:
                LOG(FATAL) << "Unknown format " << static_cast<int32_t>(format);
                return -1;  // Avoid compiler warning about missing return value
        }
    }

    inline int GetFilterTensorInputChannelDimIndex(int num_dims, FilterTensorFormat format){
        switch (format) {
            case FilterTensorFormat::OIHW:
                return num_dims-3;
            case FilterTensorFormat::HWIO:
                return num_dims-2;
            case FilterTensorFormat::HWI4O4:
                return num_dims-3;
            default:
                LOG(FATAL) << "Unknown format " << static_cast<int32_t>(format);
                return -1;  // Avoid compiler warning about missing return value
        }
    }

    inline int GetFilterTensorOutputChannelDimIndex(int num_dims, FilterTensorFormat format){
        switch (format) {
            case FilterTensorFormat::OIHW:
                return num_dims-4;
            case FilterTensorFormat::HWIO:
                return num_dims-1;
            case FilterTensorFormat::HWI4O4:
                return num_dims-4;
            default:
                LOG(FATAL) << "Unknown format " << static_cast<int32_t>(format);
                return -1;  // Avoid compiler warning about missing return value
        }
    }
}//namespace


#endif
