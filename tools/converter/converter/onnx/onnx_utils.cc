#include "onnx/onnx_utils.h"
#include <glog/logging.h>
#include <sstream>



void ParseAttrValueToString(const onnx::AttributeProto& attr,
        std::string* str){
    onnx::AttributeProto::AttributeType type = attr.type();
    std::stringstream ss;
    switch(type){
        case onnx::AttributeProto::INTS:
            for(int i=0;i<attr.ints_size();++i){
                ss<<attr.ints(i)<<" ";
            }
            break;
        case onnx::AttributeProto::INT:
            ss<<attr.i();
            break;
        case onnx::AttributeProto::FLOAT:
            ss<<attr.f();
            break;
        case onnx::AttributeProto::FLOATS:
            for(int i=0;i<attr.ints_size();++i){
                ss<<attr.floats(i)<<" ";
            }
            break;
        default:
            LOG(WARNING)<<"unknown types: "<<type;
    }
    *str = ss.str();
}
