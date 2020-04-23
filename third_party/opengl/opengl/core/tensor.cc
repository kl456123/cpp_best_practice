#include "tensor.h"


namespace opengl{
    Tensor::~Tensor(){}

    template<>
        Tensor::Tensor(float* data, DataType dtype, int num){
            size_ = num*sizeof(float);
            dtype_=dtype;
            host_ = data;
            auto temp_shape = std::vector<int>({num});
            shape_ = TensorShape(temp_shape);
        }
}//namespace opengl
