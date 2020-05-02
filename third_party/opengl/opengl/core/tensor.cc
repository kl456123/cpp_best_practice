#include "opengl/core/tensor.h"


namespace opengl{
    Tensor::~Tensor(){}

    template<>
        Tensor::Tensor(float* data, DataType dtype, INTLIST shape)
        :shape_(shape),dtype_(dtype),mem_type_(HOST_MEMORY){
            size_ = shape_.num_elements()*sizeof(float);
            host_ = data;
        }
}//namespace opengl
