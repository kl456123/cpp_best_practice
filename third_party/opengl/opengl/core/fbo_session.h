#ifndef OPENGL_CORE_FBO_SESSION_H_
#define OPENGL_CORE_FBO_SESSION_H_

#include "opengl/core/types.h"
#include "opengl/core/context.h"
#include "opengl/core/dlxnet.pb.h"

namespace opengl{
    class Env;
    class StepStats;
    class StepStatsCollector;

    class FBOSession{
        public:
            FBOSession(Context* context);
            FBOSession():FBOSession(GetContext()){}
            virtual ~FBOSession();

            /*! use inputs to allocate tensor, prepare all memory
             * to run late
             */
            void Setup(const NamedTensorList& inputs_cpu,
                    StepStatsCollector* step_collector);

            /*!
             * Draw texture to framebuffer, then
             */
            void Run(const NamedTensorList& inputs_cpu);
            void Run(const NamedTensorList& inputs_cpu,
                    StepStats* step_stats);

            // load graph from literal in memory
            void LoadGraph(const ::dlxnet::ModelProto& model_proto);

            void LoadGraph(const ::dlxnet::ModelProto&& model_proto);

            // load graph from protobuf binary in disk
            void LoadGraph(std::string model_path);

            void GetOutputs(const TensorNameList& output_names,
                    const StringList& output_dformats, TensorList* outputs);

            std::string DebugString()const;
            Context* context()const{
                return context_;
            }

            bool IsONNX()const{
                return model_->producer_name()=="ONNX";
            }

        private:

            void AllocateTensor(const TensorShapeList& shapes, TensorList& tensors);

            // reorder all nodes in nodes_ topologically
            void TopologicalSort();

            bool CheckKernelReady(const Kernel* kernel);
            void UpdateKernelReady(const Kernel* kernel);

            Tensor* FindTensorByName(const std::string& name);
            // caller does not own it
            Tensor* FindTensorById(const int id);

            Context* context_;
            OwnedKernelList kernels_;

            dlxnet::ModelProto* model_;

            // contains all tensors used in the session
            // may be some slots are null due to that pruned and optimization
            OwnedTensorList total_tensors_;

            // check session is freezed or not
            // note that when graph is freezed, session can be called multiple times
            bool finalized_ = false;

            bool graph_created_= false;

            // map from tensor name to index in total_tensors_
            NamedIndex tensor_name_index_;

            std::vector<bool> ready_;

            friend class Kernel;
            Env* env_;
    };
}//namespace opengl


#endif
