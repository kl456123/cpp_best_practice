#ifndef TENSOR_H_
#define TENSOR_H_
#include <vector>
#include <memory>


class Tensor{
    struct TensorInfo{
        int dims;
    };
    public:
        Tensor();
        virtual ~Tensor(){}


        static Tensor* Create(std::vector<int>tensor_shape);

    private:
        std::shared_ptr<TensorInfo> mTensorInfo;
};

#endif
