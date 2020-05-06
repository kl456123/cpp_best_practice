#include <iostream>
#include <string>
#include <vector>
#include "core/op_converter.h"
#include "onnx/onnx_utils.h"
#include "onnx.pb.h"



// no comma
// DEFINE_OP_CONVERTER(ConvOpConverter)


class ConvOpConverter: public OpConverter{
    public:
        ConvOpConverter(){}
        virtual ~ConvOpConverter(){}
        virtual void Run(Node* dst_node, const void* src_node)override;
};


void ConvOpConverter::Run(Node* dst_node, const void* src_node){
    dst_node->set_name("conv");
    dst_node->set_type("conv");
    dst_node->mutable_attr();
    const auto src_node_onnx = reinterpret_cast<const onnx::NodeProto*>(src_node);
    for(int i=0;i<src_node_onnx->attribute_size();i++){
        const onnx::AttributeProto& attr = src_node_onnx->attribute(i);
        // parse attr according its types and names
        // five attrs in total, they are
        // 1. dilation
        // 2. groups
        // 3. kernel_shape
        // 4. pads
        // 5. strides
        std::string pieces;
        ParseAttrValueToString(attr, &pieces);

        // print name with its value
        LOG(INFO)<<attr.name()<<": "<< pieces;
    }
}


REGISTER_OP_WITH_NAME(ConvOpConverter, "Conv");





