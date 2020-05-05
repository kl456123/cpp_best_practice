#ifndef OPENGL_CORE_FBO_SESSION_H_
#define OPENGL_CORE_FBO_SESSION_H_

#include "opengl/core/types.h"
#include "opengl/core/context.h"

namespace opengl{
    class FBOSession{
        public:
            FBOSession(Context* context);
            FBOSession():FBOSession(new Context){}
            virtual ~FBOSession();

            /*!
             *
             */
            void Setup(TensorList inputs);

            /*!
             * Draw texture to framebuffer, then
             */
            void Run();

            void LoadGraph(StringList kernel_names);

            void GetOutputs(TensorList outputs);

        private:
            void CreateVertexShader();
            GLuint CreateShader(GLenum shader_kind, const char *shader_src);

            void AllocateTensor(const TensorShapeList& shapes, TensorList& tensors);
            void Download(Tensor* tensor);
            void Upload(Tensor* cpu_tensor, Tensor* device_tensor);

            GLuint vertex_shader_;
            Context* context_;
            KernelList kernels_;

            // store their output here for each op kernel
            std::vector<TensorList> output_tensors_;

            // store their input here for each op kernel
            std::vector<TensorList> input_tensors_;
    };
}//namespace opengl


#endif
