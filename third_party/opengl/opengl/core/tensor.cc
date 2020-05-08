#include "opengl/core/tensor.h"


namespace opengl{
    Tensor::~Tensor(){}

    template<>
        Tensor::Tensor(DataType dtype, INTLIST shape, float* data)
        :shape_(shape),dtype_(dtype),mem_type_(HOST_MEMORY){
            size_ = shape_.num_elements()*sizeof(float);
            host_ = data;

            // make sure the length of shape equals to 4
            if(shape_.dims_size()<4){
                for(int i=0;i<4-shape_.dims_size();++i){
                    shape_.insert_dim(0, 1);
                }
            }

            initialized_=true;
        }
}//namespace opengl
