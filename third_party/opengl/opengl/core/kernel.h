#ifndef OPENGL_CORE_KERNEL_H_
#define OPENGL_CORE_KERNEL_H_
#include <string>

#include "opengl/core/types.h"

namespace opengl{
    class Program;
    class Tensor;
    class Context;

    class Kernel{
        public:
            Kernel(Context* context);
            virtual ~Kernel();

            virtual void SetupProgram(GLuint vertex_shader);
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
            // attach output tensor to the target(fbo)
            // used in compute function of subclass
            void SetFrameBuffer(TensorList& outputs);

            void SetVertexShader();

            // kernel program(opencl) or shader(opengl)
            Program* program_;

            // opengl driver, it wrapping all API about platform(opengl or opencl)
            Context* context_;

            // filename of kernel source file
            std::string kernel_fname_;

            // global works size and local work size
            unsigned long work_sizes_[3];

            // output target in each kernel
            GLuint frame_buffer_;
    };
}//namespace opengl


#endif
