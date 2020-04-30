#include "tensor.h"


namespace opengl{
    Tensor::~Tensor(){}

    template<>
        Tensor::Tensor(float* data, DataType dtype, INTLIST shape)
        :shape_(shape),dtype_(dtype){
            size_ = shape_.num_elements()*sizeof(float);
            host_ = data;
            mem_type_ = HOST_MEMORY;
        }
}//namespace opengl
