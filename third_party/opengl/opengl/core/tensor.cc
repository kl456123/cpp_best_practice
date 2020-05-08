#include "opengl/core/tensor.h"


namespace opengl{
    Tensor::~Tensor(){}

    template<>
        Tensor::Tensor(DataType dtype, INTLIST shape, float* data)
        :shape_(shape),dtype_(dtype),mem_type_(HOST_MEMORY){
            // make sure the length of shape equals to 4
            int dims_size = shape_.dims_size();
            if(dims_size<4){
                for(int i=0;i<4-dims_size;++i){
                    shape_.insert_dim(0, 1);
                }
            }

            dformat_=dlxnet::TensorProto::NHWC;
            size_ = shape_.num_elements()*sizeof(float);
            host_ = data;

            initialized_=true;
        }

    Tensor::Tensor(const dlxnet::TensorProto& tensor_proto){
        // get tensor shape
        for(auto& dim:tensor_proto.dims()){
            shape_.add_dim(dim);
        }
        // get tensor data
        switch(tensor_proto.data_type()){
            case dlxnet::TensorProto::FLOAT32:
                {
                    float* target_data = new float[num_elements()];
                    host_ = static_cast<void*>(target_data);
                    dtype_ = DT_FLOAT;
                    for(int i=0;i<num_elements();++i){
                        target_data[i] = tensor_proto.float_data(i);
                    }
                    break;
                }

            case dlxnet::TensorProto::INT32:
                {
                    int* target_data = new int[num_elements()];
                    host_ = static_cast<void*>(target_data);
                    dtype_ = DT_INT;
                    for(int i=0;i<num_elements();++i){
                        target_data[i] = tensor_proto.int32_data(i);
                    }
                    break;
                }

            default:
                LOG(FATAL)<<"unsupported const type: "<<tensor_proto.data_type();
        }
        initialized_=true;
        mem_type_ = HOST_MEMORY;
        dformat_=dlxnet::TensorProto::NHWC;
    }
}//namespace opengl
