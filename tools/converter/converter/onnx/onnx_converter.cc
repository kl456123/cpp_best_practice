#include <string>
#include <unordered_map>

#include "onnx/onnx_converter.h"
#include "core/utils.h"
#include "core/op_converter.h"



ONNXConverter::ONNXConverter(const ConverterConfig config)
    :Converter(config){}

    void ONNXConverter::MakeTensorFromProto(const onnx::TensorProto&, TensorProto*){
    }

void ONNXConverter::Run(){
    // load onnx proto
    onnx::ModelProto model_proto;
    bool success = utils::ONNXReadProtoFromBinary(converter_config_.src_model_path.c_str(), &model_proto);
    CHECK(success)<<"load onnx model failed!";
    const auto& graph_proto = model_proto.graph();
    const int node_counts = graph_proto.node_size();

    LOG(INFO)<<"node_counts: "<<node_counts;

    // The goal is to populate model proto

    // populate graph
    Graph* graph = model_->mutable_graph();
    // map from tensor name to tensor index
    std::unordered_map<std::string, int> total_tensor_names;

    // insert input tensor to total_tensor_names first
    const int input_tensor_count = graph_proto.input_size();
    for(int i=0;i<input_tensor_count;++i){
        total_tensor_names.insert({graph_proto.input(i).name(), i});
    }

    // constant tensor map
    // help to find const tensor more quickly
    std::unordered_map<std::string, const onnx::TensorProto*>
        constant_tensor_map;
    const int constant_tensor_count = graph_proto.initializer_size();
    for(int i=0;i<constant_tensor_count;++i){
        auto& initializer = graph_proto.initializer(i);
        constant_tensor_map.insert({initializer.name(), &initializer});
    }


    for(int i=0;i<node_counts;i++){
        // dispatch each node handlers
        const auto& node_proto = graph_proto.node(i);
        const auto& op_type = node_proto.op_type();
        LOG(INFO)<<"op_type: "<<op_type;

        // find op according to its name
        auto op_conveter_registry = Registry<OpConverter>::Global();
        OpConverter* op_converter=nullptr;
        op_conveter_registry->LookUp(op_type, &op_converter);

        // handle with its input first
        // check if they contain constant tensor or not
        for(int i=0;i<node_proto.input_size();++i){
            auto& input_name = node_proto.input(i);
            // find input in constant map and total_tensor_names first
            if(total_tensor_names.find(input_name)!=total_tensor_names.end()){
                // already inserted, skip it
                continue;
            }
            auto iter = constant_tensor_map.find(input_name);
            if(iter!=constant_tensor_map.end()){
                // constant tensor
                Node* node_ptr = graph->add_node();
                node_ptr->set_name(iter->first);
                node_ptr->set_type("Const");

                // convert data

                TensorProto* tensor = node_ptr->mutable_attr()
                    ->mutable_const_attr()->mutable_value();
                MakeTensorFromProto(*iter->second, tensor);
                total_tensor_names.insert({input_name, total_tensor_names.size()});
            }
        }

        // then handle node according to its type

        if(op_converter==nullptr){
            LOG(WARNING)<<"Cannot find Type: "<<op_type;
            continue;
        }else{
            Node* node_ptr = graph->add_node();
            // populate node
            op_converter->Run(node_ptr, &node_proto);
        }
    }
    LOG(INFO)<<"ONNXConverter Done!";
}


