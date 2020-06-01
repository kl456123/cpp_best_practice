#include "core/op_converter.h"
#include "onnx/onnx_utils.h"
#include "onnx.pb.h"

DECLARE_OP_CONVERTER(Const);

namespace {
    void MakeTensorFromProto(const onnx::TensorProto& onnx_tensor,
            dlxnet::TensorProto* dlcl_tensor){
        // switch data transfer according to its data type
        auto data_type = static_cast<onnx::TensorProto::DataType>(
                onnx_tensor.data_type());
        // handle tensor shape
        size_t num_elements =1;
        size_t dim_size = onnx_tensor.dims().size();
        for (int i = 0; i < dim_size; ++i) {
            dlcl_tensor->add_dims(onnx_tensor.dims(i));
            num_elements  *= onnx_tensor.dims(i);
        }

        // handle tensor value
        const void* tensor_content = onnx_tensor.raw_data().data();
        switch(data_type){
            case onnx::TensorProto::FLOAT:
                {
                    auto source = (float*)tensor_content;
                    for(int i=0;i<num_elements;++i){
                        dlcl_tensor->add_float_data(source[i]);
                    }
                    dlcl_tensor->set_data_type(dlxnet::TensorProto::FLOAT32);
                    break;
                }
            case onnx::TensorProto::INT64:
                {
                    auto source = (int32_t*)tensor_content;
                    for(int i=0;i<num_elements;++i){
                        dlcl_tensor->add_int32_data(source[i]);
                    }
                    dlcl_tensor->set_data_type(dlxnet::TensorProto::INT32);
                    break;
                }
            default:
                LOG(WARNING)<<"unsupported data type when converting tensor "
                    <<data_type;
        }

    }
}// namespace

void  ConstOpConverter::SetTensorInfo(dlxnet::TensorProto* dlcl_tensor,
        int tensor_index){
    LOG(FATAL)<<"Cannot set tensor info for const due to the only one input tensor";
}

void ConstOpConverter::Run(dlxnet::NodeProto* dst_node, const void* src_node){
    dlxnet::ConstAttribute* dst_attr = dst_node->mutable_attr()->mutable_const_attr();
    const auto src_node_onnx = reinterpret_cast<const onnx::NodeProto*>(src_node);
    std::string res;
    for(int i=0;i<src_node_onnx->attribute_size();i++){
        const onnx::AttributeProto& attr = src_node_onnx->attribute(i);
        if(attr.name()=="value"){
            dlxnet::TensorProto* tensor = dst_attr->mutable_value();
            MakeTensorFromProto(attr.t(), tensor);
        }else{
            ParseAttrValueToString(attr, &res);
            LOG(INFO)<<res;
        }
    }
}


REGISTER_OP_WITH_NAME(ConstOpConverter, "Constant");
