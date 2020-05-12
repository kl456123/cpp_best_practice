#ifndef OPENGL_CORE_FBO_SESSION_H_
#define OPENGL_CORE_FBO_SESSION_H_

#include "opengl/core/types.h"
#include "opengl/core/context.h"
#include "opengl/core/dlxnet.pb.h"

namespace opengl{
    class FBOSession{
        public:
            FBOSession(Context* context);
            FBOSession():FBOSession(GetContext()){}
            virtual ~FBOSession();

            /*! use inputs to allocate tensor, prepare all memory
             * to run late
             */
            void Setup(const NamedTensorList& inputs_cpu);

            /*!
             * Draw texture to framebuffer, then
             */
            void Run();

            // load graph from literal in memory
            void LoadGraph(StringList kernel_names);

            // load graph from protobuf binary in disk
            void LoadGraph(std::string model_path);

            void GetOutputs(const TensorNameList& output_names,
                   const StringList& output_dformats, TensorList* outputs);

            std::string DebugString()const;

        private:
            void CreateVertexShader();
            GLuint CreateShader(GLenum shader_kind, const char *shader_src);

            // attach output tensor to the target(fbo)
            // used in compute function of subclass
            void SetupFrameBuffer();

            void AllocateTensor(const TensorShapeList& shapes, TensorList& tensors);

            // reorder all nodes in nodes_ topologically
            void TopologicalSort();

            GLuint vertex_shader_;
            Context* context_;
            KernelList kernels_;

            dlxnet::ModelProto* model_;

            // contains all tensors used in the session
            // may be some slots are null due to that pruned and optimization
            std::vector<Tensor*> total_tensors_;

            // contains all nodes used in the session
            // including constant node
            std::vector<Kernel*> nodes_;

            // check session is freezed or not
            // note that when graph is freezed, session can be called multiple times
            bool finalized_ = false;

            // map from tensor name to index in total_tensors_
            NamedIndex tensor_name_index_;

            // output target in each kernel
            GLuint frame_buffer_;
    };
}//namespace opengl


#endif
