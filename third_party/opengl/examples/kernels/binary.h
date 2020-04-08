#ifndef KERNELS_BINARY_H_
#define KERNELS_BINARY_H_
#include <vector>


class Program;
class Tensor;
class Context;

typedef std::vector<Tensor*> TensorList;

class BinaryKernel{
    public:
        BinaryKernel(Context* context);
        virtual void Compute(TensorList& inputs, TensorList& outputs);
        virtual ~BinaryKernel();
    private:
        Program* program_;
        Context* context_;
        unsigned long work_sizes_[3];
};


#endif
