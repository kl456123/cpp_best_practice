#ifndef OPENGL_NN_KERNELS_TRANSPOSE_H_
#define OPENGL_NN_KERNELS_TRANSPOSE_H_
#include <vector>
#include "opengl/core/types.h"
#include "opengl/core/kernel.h"


namespace opengl{
    class Context;

        class TransposeKernel: public Kernel{
            public:
                TransposeKernel(Context* context);
                virtual void Compute(TensorList& inputs, TensorList& outputs);
                virtual void InferOutputShape(TensorShapeList& inputs,
                        TensorShapeList& outputs);
                virtual void SetupAttr(const dlxnet::Attribute& attr);
                virtual ~TransposeKernel();
            private:
                std::vector<int> perm_;

        };
}//namespace opengl

#endif
