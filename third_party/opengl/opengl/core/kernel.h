#ifndef OPENGL_CORE_KERNEL_H_
#define OPENGL_CORE_KERNEL_H_
#include "opengl/core/types.h"

namespace opengl{
    class Program;
    class Tensor;
    class Context;

    class Kernel{
        public:
            Kernel(Context* context);
            virtual ~Kernel();
            /*!
             * Run Kernel, do computation actually
             */
            virtual void Compute(TensorList& inputs, TensorList& outputs)=0;

            /*!
             * Compute output shapes according to their input tensor shape
             */
            virtual void InferOutputShape(TensorShapeList& inputs,
                    TensorShapeList& outputs)=0;
        protected:
            // kernel program(opencl) or shader(opengl)
            Program* program_;

            // opengl driver, it wrapping all API about platform(opengl or opencl)
            Context* context_;

            // global works size and local work size
            unsigned long work_sizes_[3];
    };
}//namespace opengl


#endif
