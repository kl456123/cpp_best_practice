#include "opengl/core/tensor_format.h"
#include "opengl/utils/macros.h"

namespace opengl{
    int GetChannel(const IntList& shape, DataFormat dformat){
        CHECK_GE(shape.size(), 4);
        if(dformat==::dlxnet::TensorProto::NHWC){
            return shape[3];
        }
        if(dformat==::dlxnet::TensorProto::NCHW){
            return shape[1];
        }

        if(dformat==::dlxnet::TensorProto::NHWC4){
            return shape[3];
        }

        if(dformat==::dlxnet::TensorProto::HWN4C4){
            return shape[3];
        }
        LOG(FATAL)<<"dformat: "<< dformat<<" is not correct";
    }

    int GetBatch(const IntList& shape,
            DataFormat dformat){
        CHECK_GE(shape.size(), 4);
        if(dformat==::dlxnet::TensorProto::NHWC){
            return shape[0];
        }
        if(dformat==::dlxnet::TensorProto::NCHW){
            return shape[0];
        }

        if(dformat==::dlxnet::TensorProto::NHWC4){
            return shape[0];
        }

        if(dformat==::dlxnet::TensorProto::HWN4C4){
            return shape[2];
        }
        LOG(FATAL)<<"dformat: "<< dformat<<" is not correct";
    }

    int GetWidth(const IntList& shape,
            DataFormat dformat){
        CHECK_GE(shape.size(), 4);
        if(dformat==::dlxnet::TensorProto::NHWC){
            return shape[2];
        }
        if(dformat==::dlxnet::TensorProto::NCHW){
            return shape[3];
        }

        if(dformat==::dlxnet::TensorProto::NHWC4){
            return shape[2];
        }

        if(dformat==::dlxnet::TensorProto::HWN4C4){
            return shape[1];
        }
        LOG(FATAL)<<"dformat: "<< dformat<<" is not correct";
    }

    int GetHeight(const IntList& shape,
            DataFormat dformat){
        CHECK_GE(shape.size(), 4);
        if(dformat==::dlxnet::TensorProto::NHWC){
            return shape[1];
        }
        if(dformat==::dlxnet::TensorProto::NCHW){
            return shape[2];
        }

        if(dformat==::dlxnet::TensorProto::NHWC4){
            return shape[1];
        }

        if(dformat==::dlxnet::TensorProto::HWN4C4){
            return shape[0];
        }
        LOG(FATAL)<<"dformat: "<< dformat<<" is not correct";
    }

    IntList MakeTensorShape(const int batch, const int height,
            const int width, const int channels, DataFormat dformat){
        if(dformat==::dlxnet::TensorProto::NHWC){
            return {batch, height, width, channels};
        }
        if(dformat==::dlxnet::TensorProto::NCHW){
            return {batch, channels, height, width};
        }

        if(dformat==::dlxnet::TensorProto::NHWC4){
            return {batch, height, width, UP_DIV(channels, 4), 4};
        }

        if(dformat==::dlxnet::TensorProto::HWN4C4){
            return {height, width, UP_DIV(batch, 4), UP_DIV(channels, 4), 4, 4};
        }
        LOG(FATAL)<<"dformat: "<< dformat<<" is not correct";
    }

    IntList TensorShapeFromFormat(DataFormat dst_format,
            const IntList& src_shape, DataFormat src_format) {
        if (src_format == dst_format) {
            return src_shape;
        }
        auto channels = GetChannel(src_shape, src_format);
        auto batch = GetBatch(src_shape, src_format);
        auto width = GetWidth(src_shape, src_format);
        auto height = GetHeight(src_shape, src_format);

        // compose them according to dst_format
        return MakeTensorShape(batch, height, width, channels, dst_format);
    }

    IntList MakeTextureShape(const IntList shape, DataFormat dformat){
        // make sure the correct dformat
        CHECK(dformat==::dlxnet::TensorProto::NHWC4
                ||::dlxnet::TensorProto::HWN4C4);
        if(dformat==::dlxnet::TensorProto::NHWC4){
            CHECK_EQ(shape.size(), 5);
            return {shape[2]*shape[3], shape[0]*shape[1], 4};
        }
        if(dformat==::dlxnet::TensorProto::HWN4C4){
            CHECK_EQ(shape.size(), 6);
            return {shape[3]*4, shape[0]*shape[1]*shape[2], 4};
        }
        LOG(FATAL)<<"dformat: "<< dformat<<" is not correct";
    }

    DataFormat StrToFormat(std::string dformat_str){
        DataFormat dformat;
        if(dformat_str=="NHWC"){
            dformat = dlxnet::TensorProto::NHWC;
        }else if(dformat_str=="NCHW"){
            dformat = dlxnet::TensorProto::NCHW;
        }else if(dformat_str=="ANY"){
            dformat = dlxnet::TensorProto::ANY;
        }else if(dformat_str=="NHWC4"){
            dformat = dlxnet::TensorProto::NHWC4;
        }else if(dformat_str=="HWN4C4"){
            dformat = dlxnet::TensorProto::HWN4C4;
        }else{
            LOG(FATAL)<<"unsupported dformat_str: "<<dformat_str;
        }
        return dformat;
    }


    std::string FormatToStr(DataFormat dformat){
        if(dformat==dlxnet::TensorProto::NHWC){
            return "NHWC";
        }else{
            LOG(FATAL)<<"unsupported dformat: "<<dformat;
        }
    }


    DataFormat FormatToStride4(DataFormat dformat){
        if(dformat==dlxnet::TensorProto::NHWC){
            return dlxnet::TensorProto::NHWC4;
        }
        if(dformat==dlxnet::TensorProto::ANY){
            return dlxnet::TensorProto::ANY4;
        }
        LOG(FATAL)<<"unsupported dformat: "<<dformat;
    }
}//namespace opengl
