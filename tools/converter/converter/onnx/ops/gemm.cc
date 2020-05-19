#include "core/op_converter.h"
#include "onnx/onnx_utils.h"
#include "onnx.pb.h"

DECLARE_OP_CONVERTER(Gemm);

void GemmOpConverter::SetTensorInfo(dlxnet::TensorProto* dlcl_tensor,
        int tensor_index){
    // Y = WX+B
    CHECK(tensor_index==1||tensor_index==2);

    if(tensor_index==1){
        // 1. fc.weight (N_out, N_in)
        CHECK_EQ(dlcl_tensor->dims_size(), 2);
        // set shape
        const int n_out = dlcl_tensor->dims(0);
        const int n_in = dlcl_tensor->dims(1);

        dlcl_tensor->clear_dims();
        dlcl_tensor->add_dims(n_out);
        dlcl_tensor->add_dims(n_in);
        // set spatial dim to 1x1
        dlcl_tensor->add_dims(1);
        dlcl_tensor->add_dims(1);

        // set format
        dlcl_tensor->set_target_data_format(dlxnet::TensorProto::HWN4C4);
    }else{
        // 2. fc.bias(N_out)
        CHECK_EQ(dlcl_tensor->dims_size(), 1);
        const int n_out = dlcl_tensor->dims(0);

        dlcl_tensor->clear_dims();
        dlcl_tensor->add_dims(1);
        dlcl_tensor->add_dims(n_out);
        dlcl_tensor->add_dims(1);
        dlcl_tensor->add_dims(1);
        // set format
        dlcl_tensor->set_target_data_format(dlxnet::TensorProto::NHWC4);
    }
    dlcl_tensor->set_data_format(dlxnet::TensorProto::NCHW);
}

void GemmOpConverter::Run(dlxnet::NodeProto* dst_node, const void* src_node){
    dlxnet::GemmAttribute* dst_attr = dst_node->mutable_attr()->mutable_gemm_attr();
    const auto src_node_onnx = reinterpret_cast<const onnx::NodeProto*>(src_node);
    std::string res;
    for(int i=0;i<src_node_onnx->attribute_size();i++){
        const onnx::AttributeProto& attr = src_node_onnx->attribute(i);
        if(attr.name()=="alpha"){
            dst_attr->set_alpha(attr.f());
        }else if(attr.name()=="beta"){
            dst_attr->set_beta(attr.f());
        }else if(attr.name()=="transB"){
            dst_attr->set_transb(attr.i());
        }else{
            ParseAttrValueToString(attr, &res);
            LOG(INFO)<<res;
        }

    }
}


REGISTER_OP_WITH_NAME(GemmOpConverter, "Gemm");
