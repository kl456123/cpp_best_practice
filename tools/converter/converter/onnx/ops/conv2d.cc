#include <iostream>
#include <string>
#include <vector>
#include "core/op_converter.h"
#include "onnx/onnx_utils.h"
#include "onnx.pb.h"



DECLARE_OP_CONVERTER(Conv);

void ConvOpConverter::SetTensorInfo(dlxnet::TensorProto* dlcl_tensor, int tensor_index){
    // only filter and bias can be set
    CHECK(tensor_index==1||tensor_index==2);

    if(tensor_index==1){
        // filter
        dlcl_tensor->set_target_data_format(dlxnet::TensorProto::HWN4C4);
        // set shape
        // no need to change filter shape due to that it has
        // nchw(n_out, n_in, h, w) dformat already
        CHECK_EQ(dlcl_tensor->dims_size(), 4);
    }else{
        // bias, shape(n_out)
        dlcl_tensor->set_target_data_format(dlxnet::TensorProto::NHWC4);
        CHECK_EQ(dlcl_tensor->dims_size(), 1);
        const int n_out = dlcl_tensor->dims(0);

        // set shape
        // TODO(breakpoint)simpliy it
        dlcl_tensor->clear_dims();
        dlcl_tensor->add_dims(1);
        dlcl_tensor->add_dims(n_out);
        dlcl_tensor->add_dims(1);
        dlcl_tensor->add_dims(1);
    }
    dlcl_tensor->set_data_format(dlxnet::TensorProto::NCHW);
}


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





