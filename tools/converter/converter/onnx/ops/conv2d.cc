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
        virtual void Run(dlxnet::NodeProto* dst_node, const void* src_node)override;
};


void ConvOpConverter::Run(dlxnet::NodeProto* dst_node, const void* src_node){
    dlxnet::Conv2dAttribute* dst_attr = dst_node->mutable_attr()->mutable_conv2d_attr();
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
        if(attr.name()=="kernel_shape"){
            for(int i=0;i<attr.ints_size();++i){
                dst_attr->add_kernel_shape(attr.ints(i));
            }
        }else if(attr.name()=="strides"){
            for(int i=0;i<attr.ints_size();++i){
                dst_attr->add_strides(attr.ints(i));
            }
        }else if(attr.name()=="pads"){
            for(int i=0;i<attr.ints_size();++i){
                dst_attr->add_pads(attr.ints(i));
            }
        }
    }
}


REGISTER_OP_WITH_NAME(ConvOpConverter, "Conv");





