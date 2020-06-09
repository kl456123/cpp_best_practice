#ifndef OPENGL_CORE_FUNCTOR_H_
#define OPENGL_CORE_FUNCTOR_H_

namespace opengl{
    class Tensor;
    class Context;
    namespace functor{
        // some commonly used kernel functions, but we dont consider them as kernel to simpliy
        // logic, just make it as a functor.
        // Note that `Functor` is a struct overrided operator() to be called more easily.
        struct ConvertTensorNHWC4ToANY4{
            void operator()(Context* ctx, const Tensor* src_tensor, Tensor* dst_tensor);
        };
    }//namespace functor
}// namespace opengl


#endif
