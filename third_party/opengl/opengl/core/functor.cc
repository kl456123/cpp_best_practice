#include "opengl/core/functor.h"
#include "opengl/core/tensor.h"
#include "opengl/core/context.h"
#include "opengl/core/program.h"
#include <glog/logging.h>

namespace opengl{
    namespace internal{
        IntList AmendShape(const IntList& shape){
            CHECK_LE(shape.size(), 4);
            const int remain_dims = 4-shape.size();
            IntList amended_shape = shape;
            for(int i=0;i<remain_dims;++i){
                amended_shape.insert(amended_shape.begin(), 1);
            }
            return amended_shape;
        }

        void RunOpenGLProgram(const std::string& kernel_fname, Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            // common used for copy from host to device
            CHECK(!src_tensor->is_host());
            CHECK(!dst_tensor->is_host());

            auto program = std::unique_ptr<Program>(ctx->CreateProgram(kernel_fname));
            // activate it before use
            program->Activate();

            program->SetRetVal({dst_tensor});

            // set input
            auto src_texture = src_tensor->device<Texture>();
            program->set_vec4i("output_shape", AmendShape(dst_tensor->shape()));
            // input
            {
                program->set_image2D("input_image", src_texture->id(),  0);
                OPENGL_CHECK_ERROR;
            }
            program->Run();
        }
    }//namespace internal
    namespace functor{
        void ConvertTensorNHWC4ToANY4::operator()(Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            internal::RunOpenGLProgram("../opengl/nn/glsl/nhwc4_to_any4.glsl",
                    ctx, src_tensor, dst_tensor);
        }

        void ConvertTensorANYToANY4::operator()(Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            internal::RunOpenGLProgram("../opengl/nn/glsl/any_to_any4.glsl",
                    ctx, src_tensor, dst_tensor);

        }

        void ConvertTensorANY4ToANY::operator()(Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            internal::RunOpenGLProgram("../opengl/nn/glsl/any4_to_any.glsl",
                    ctx, src_tensor, dst_tensor);
        }

        void ConvertTensorANYToNHWC4::operator()(Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            internal::RunOpenGLProgram("../opengl/nn/glsl/any_to_nhwc4.glsl",
                    ctx, src_tensor, dst_tensor);
        };

        void ConvertTensorNCHWToHWN4C4::operator()(Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            internal::RunOpenGLProgram("../opengl/nn/glsl/nchw_to_hwn4c4.glsl",
                    ctx, src_tensor, dst_tensor);
        }

        void ConvertTensorTest::operator()(Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            internal::RunOpenGLProgram("../opengl/nn/glsl/layout_test.glsl",
                    ctx, src_tensor, dst_tensor);
        };

        void ConvertTensorNHWC4ToANY::operator()(Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            internal::RunOpenGLProgram("../opengl/nn/glsl/nhwc4_to_any.glsl",
                    ctx, src_tensor, dst_tensor);
        };

        void ConvertTensorHWN4C4ToNCHW::operator()(Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            internal::RunOpenGLProgram("../opengl/nn/glsl/hwn4c4_to_nchw.glsl",
                    ctx, src_tensor, dst_tensor);
        };
    }//namespace functor
}//namespace opengl
