#include "core/op_converter.h"
#include "onnx/onnx_utils.h"
#include "onnx.pb.h"

DECLARE_OP_CONVERTER(Gemm);

void GemmOpConverter::SetTensorInfo(dlxnet::TensorProto* dlcl_tensor,
        int tensor_index){
    LOG(INFO)<<"in Gemm index: "<<tensor_index;
}

void GemmOpConverter::Run(dlxnet::NodeProto* dst_node, const void* src_node){
    dlxnet::GemmAttribute* dst_attr = dst_node->mutable_attr()->mutable_gemm_attr();
    const auto src_node_onnx = reinterpret_cast<const onnx::NodeProto*>(src_node);
    std::string res;
    for(int i=0;i<src_node_onnx->attribute_size();i++){
        const onnx::AttributeProto& attr = src_node_onnx->attribute(i);
        ParseAttrValueToString(attr, &res);
        LOG(INFO)<<res;
    }
}


REGISTER_OP_WITH_NAME(GemmOpConverter, "Gemm");
