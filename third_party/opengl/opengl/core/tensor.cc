#include "opengl/core/tensor.h"


namespace opengl{
    Tensor::~Tensor(){}

    template<>
        Tensor::Tensor(DataType dtype, INTLIST shape, float* data)
        :shape_(shape),dtype_(dtype),mem_type_(HOST_MEMORY){
            size_ = shape_.num_elements()*sizeof(float);
            host_ = data;

            initialized_=true;
        }
}//namespace opengl
