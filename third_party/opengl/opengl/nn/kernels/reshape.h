#ifndef OPENGL_NN_KERNELS_RESHAPE_H_
#define OPENGL_NN_KERNELS_RESHAPE_H_
#include <vector>
#include "opengl/core/types.h"
#include "opengl/core/kernel.h"


namespace opengl{
    class Context;

        class ReshapeKernel: public Kernel{
            public:
                ReshapeKernel(Context* context);
                virtual void Compute(TensorList& inputs, TensorList& outputs)override;
                virtual void InferOutputShape(TensorShapeList& inputs,
                        TensorShapeList& outputs)override{}
                virtual void InferOutputShape(const TensorList& inputs,
                    TensorShapeList& output_shapes)override;
                virtual void SetupAttr(const dlxnet::Attribute& attr)override;
                virtual ~ReshapeKernel();

        };
}//namespace opengl

#endif
