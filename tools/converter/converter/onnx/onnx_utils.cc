#include "onnx/onnx_utils.h"
#include <glog/logging.h>
#include <sstream>



void ParseAttrValueToString(const onnx::AttributeProto& attr,
        std::string* str){
    onnx::AttributeProto::AttributeType type = attr.type();
    std::stringstream ss;
    ss<<"name: "<<attr.name()<<"; ";
    ss<<"type: "<<attr.type()<<"; ";
    switch(type){
        case onnx::AttributeProto::INTS:
            for(int i=0;i<attr.ints_size();++i){
                ss<<attr.ints(i)<<", ";
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
                ss<<attr.floats(i)<<", ";
            }
            break;
        default:
            LOG(WARNING)<<"unknown types: "<<type;
    }
    *str = ss.str();
}

void ParseAttrListToString(const AttributeProtoList& attr_list, std::string* pieces){
    std::string res;
    for(int i=0;i<attr_list.size();i++){
        std::string tmp;
        ParseAttrValueToString(attr_list[i], &tmp);
        res+=tmp;
        // const onnx::AttributeProto& attr = src_node_onnx->attribute(i);
        // ParseAttrValueToString(attr, &res);
        // LOG(INFO)<<res;
    }
    *pieces = res;
}
