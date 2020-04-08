#include "tensor.h"


Tensor::~Tensor(){}

template<>
Tensor::Tensor(float* data, DataType dtype){
    dtype_=dtype;
    host_ = data;
}
