#include "opengl/core/functor.h"
#include "opengl/core/tensor.h"
#include "opengl/core/context.h"
#include "opengl/core/program.h"
#include <glog/logging.h>

namespace opengl{
    namespace functor{
        void ConvertTensorNHWC4ToANY4::operator()(Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            // copying in device is supported now
            CHECK(!src_tensor->is_host());
            CHECK(!dst_tensor->is_host());

            const std::string kernel_fname = "../opengl/nn/glsl/nhwc4_to_any4.glsl";
            auto program = std::unique_ptr<Program>(ctx->CreateProgram(kernel_fname));
            // activate it before use
            program->Activate();

            program->SetRetVal({dst_tensor});

            // set input
            auto src_texture = src_tensor->device<Texture>();
            program->set_vec4i("output_shape", dst_tensor->shape());
            // input
            {
                program->set_image2D("input_image", src_texture->id(),  0);
                OPENGL_CHECK_ERROR;
            }
            program->Run();
        }
    }//namespace functor
}//namespace opengl
