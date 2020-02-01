#include <iostream>

#include "core/utils.h"
#include "onnx/onnx_converter.h"
#include "core/op_converter.h"
#include "onnx.pb.h"
#include "dlcl.pb.h"



ONNXConverter::ONNXConverter(const ConverterConfig config):Converter(config){

}

void ONNXConverter::Run(){
    // load onnx proto
    onnx::ModelProto model_proto;
    bool success = utils::ONNXReadProtoFromBinary(converter_config_.src_model_path.c_str(), &model_proto);
    if(!success){
        std::cout<<"load onnx model failed!"<<std::endl;
        return;
    }
    const auto& graph_proto = model_proto.graph();
    const int node_counts = graph_proto.node_size();

    std::cout<<"node_counts: "<<node_counts<<std::endl;

    // The goal is to populate model proto
    Model model =  Model();
    model.set_producer_name("ONNX");
    model.set_version("0.1");
    model.set_doc_string("ignored");

    // populate graph
    Graph* graph = model.mutable_graph();

    for(int i=0;i<node_counts;i++){
        // dispatch each node handlers
        const auto& node_proto = graph_proto.node(i);
        const auto& op_type = node_proto.op_type();
        std::cout<<"op_type: "<<op_type<<std::endl;
        auto op_conveter_registry = Registry<OpConverter>::Global();
        std::string op_name = "ConvOpConverter";
        OpConverter* op_converter=nullptr;
        op_conveter_registry->LookUp(op_name, &op_converter);
        if(op_converter==nullptr){
            std::cout<<"Cannot find"<<std::endl;
            continue;
        }else{
            Node* node_ptr = graph->add_node();
            // populate node
            op_converter->Run(node_ptr, &node_proto);
        }
    }
    std::cout<<"ONNXConverter"<<std::endl;
}


