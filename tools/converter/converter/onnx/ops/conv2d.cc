#include <iostream>

#include "core/op_converter.h"
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
        auto& attr = src_node_onnx->attribute(i);
        std::cout<<"f: "<<attr.f()<<std::endl;
    }
    std::cout<<"ConvOpConverter"<<std::endl;
}


REGISTER_CLASS_OP(ConvOpConverter);





