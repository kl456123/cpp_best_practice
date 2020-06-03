#include "opengl/nn/apis/nn_ops.h"


namespace opengl{
    int AddConstNode(Scope* scope, const std::string&  name,
            const std::vector<int>& shape, DataFormat dformat){
        auto node_ptr = scope->AddNode();
        node_ptr->set_name(name);
        node_ptr->set_type("Const");
        int tensor_id = scope->AddTensor(name);
        node_ptr->add_output_index(tensor_id);
        dlxnet::TensorProto* dlcl_tensor = node_ptr->mutable_attr()
            ->mutable_const_attr()->mutable_value();

        int num_elements = 1;
        for(auto dim: shape){
            dlcl_tensor->add_dims(dim);
            num_elements  *= dim;
        }
        for(int j=0;j<num_elements;++j){
            dlcl_tensor->add_float_data(1.0*random()/RAND_MAX);
        }
        // set tensor
        dlcl_tensor->set_data_type(dlxnet::TensorProto::FLOAT32);
        dlcl_tensor->set_target_data_format(dformat);
        dlcl_tensor->set_data_format(dlxnet::TensorProto::NCHW);
        return tensor_id;

    }

    // TODO(breakpoint) change id to NodeOut class to store shape info
    // can use shape info do some things to validate
    int AddConvNode(Scope* scope, const std::string&  name, std::vector<int> input_ids,
            const Conv2dParams& conv2d_params){
        auto dlcl_node = scope->AddNode();
        // set node name with the name of the first output
        dlcl_node->set_name(name);

        dlcl_node->set_type("Conv");
        dlxnet::Conv2dAttribute* dst_attr = dlcl_node->mutable_attr()->mutable_conv2d_attr();
        for(int i=0;i<2;++i){
            dst_attr->add_kernel_shape(conv2d_params.kernel_size);
        }
        for(int i=0;i<2;++i){
            dst_attr->add_strides(conv2d_params.stride);
        }
        for(int i=0;i<4;++i){
            dst_attr->add_pads(conv2d_params.padding);
        }
        for(auto tensor_id:input_ids){
            dlcl_node->add_input_index(tensor_id);
        }
        // both node name and tensor name are the same
        int tensor_id = scope->AddTensor(name);
        dlcl_node->add_output_index(tensor_id);
        return tensor_id;
    }

    int AddInputNode(Scope* scope, std::string name){
        // add node
        scope->AddInputName(name);
        // add tensor
        return scope->AddTensor(name);
    }

    int AddShapeNode(Scope* scope, std::string name, std::vector<int> input_ids){
        auto dlcl_node = scope->AddNode();
        dlcl_node->set_name(name);
        dlcl_node->set_type("Shape");
        int tensor_id = scope->AddTensor(name);
        for(auto tensor_id:input_ids){
            dlcl_node->add_input_index(tensor_id);
        }
        dlcl_node->add_output_index(tensor_id);
        return tensor_id;
    }
}//namespace opengl
