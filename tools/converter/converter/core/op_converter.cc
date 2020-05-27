#include "core/op_converter.h"


OpConverter::OpConverter(){
}

OpConverter::~OpConverter(){
}

void OpConverter::SetTensorInfo(dlxnet::TensorProto* dlcl_tensor, int tensor_index){
    // fill shape to make four element tuple
    // fill one in the front of dim vector
    const int remain_dims =  4 - dlcl_tensor->dims_size();
    std::vector<int> dims(remain_dims, 1);
    for(auto dim: dlcl_tensor->dims()){
        dims.emplace_back(dim);
    }
    dlcl_tensor->clear_dims();
    for(int i=0;i<4;++i){
        dlcl_tensor->add_dims(dims[i]);
    }

    // set default dformat in onnx, nchw is common used by torch
    dlcl_tensor->set_data_format(dlxnet::TensorProto::NCHW);
    dlcl_tensor->set_data_format(dlxnet::TensorProto::NHWC4);
}
