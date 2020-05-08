#ifndef OPENGL_NN_KERNELS_CONST_H_
#define OPENGL_NN_KERNELS_CONST_H_

#include <vector>
#include "opengl/core/types.h"
#include "opengl/core/kernel.h"


namespace opengl{

    class Context;

    class ConstKernel: public Kernel{
        public:
            ConstKernel(Context* context);
            virtual void Compute(TensorList& inputs, TensorList& outputs);
            virtual void InferOutputShape(TensorShapeList& inputs,
                    TensorShapeList& outputs);
            virtual void SetupAttr(const dlxnet::Attribute& attr);
            virtual ~ConstKernel();
        private:
            // TODO(breakpoint) use Tensor instead
            std::vector<int> tensor_shape_;
            void* tensor_data_;

            int num_elements_;
    };
}//namespace opengl
#endif
