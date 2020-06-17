#include <sstream>

#include "opengl/core/fbo_session.h"
#include "opengl/utils/macros.h"
#include "opengl/utils/env.h"
#include "opengl/core/kernel.h"
#include "opengl/core/kernel_registry.h"
#include "opengl/utils/protobuf.h"
#include "opengl/core/tensor.h"
#include "opengl/core/tensor_format.h"
#include "opengl/core/driver.h"
#include "opengl/core/metric.h"
#include "opengl/core/step_stats_collector.h"
#include "opengl/core/step_stats.pb.h"
#include "opengl/utils/env.h"


namespace opengl{
    namespace{
        string kOpenGLDeviceName="OpenGL";
        uint64 time_stamp = 0;
        bool tracking_stats=false;

        void Start(){
            if(!tracking_stats){
                return;
            }
            CHECK_EQ(time_stamp, 0);
            OPENGL_CALL(glFinish());
            time_stamp=EnvTime::Default()->NowMicros();
        }

        void Stop(const string& event_name){
            if(!tracking_stats){
                return;
            }
            CHECK_NE(time_stamp, 0);
            OPENGL_CALL(glFinish());
            std::cout<<event_name<<": "<< (EnvTime::Default()->NowMicros()-time_stamp)*1e-3<<" ms\n";
            time_stamp = 0;
        }
    }
    void SetTrackingStats(bool flag){
        tracking_stats=flag;
    }
    namespace nodestats{
        inline int64 NowInNsec() { return EnvTime::Default()->NowNanos(); }

        void SetScheduled(NodeExecStatsInterface* stats, int64 micros) {
            if (!stats) return;
            stats->SetScheduled(micros * EnvTime::kMicrosToNanos);
        }

        void SetAllStart(NodeExecStatsInterface* stats) {
            if (!stats) return;
            // flush commonad queue first
            OPENGL_CALL(glFinish());
            stats->RecordExecutorStarted();
        }

        void SetOpStart(NodeExecStatsInterface* stats) {
            if (!stats) return;
            stats->RecordComputeStarted();
        }

        void SetOpEnd(NodeExecStatsInterface* stats) {
            if (!stats) return;
            stats->RecordComputeEnded();
        }

        void SetAllEnd(NodeExecStatsInterface* stats) {
            if (!stats) return;
            // make sure current node finished
            OPENGL_CALL(glFinish());
            stats->RecordExecutorEnded();
        }

        void SetOutput(NodeExecStatsInterface* stats, int slot, const Tensor* v) {
            if (!stats) return;
            stats->SetOutput(slot, v);
        }
    }//namespace nodestats
    FBOSession::~FBOSession(){
        // delete all tensors
    }

    void FBOSession::LoadGraph(const std::string file_path){
        auto model_proto = std::unique_ptr<::dlxnet::ModelProto>(
                new ::dlxnet::ModelProto);
        // load graph from disk
        CHECK(ReadProtoFromBinary(file_path.c_str(), model_proto.get()))
            <<"Load Graph "<<file_path <<"Failed";
        LoadGraph(*model_proto);
    }

    void FBOSession::LoadGraph(const ::dlxnet::ModelProto&& model_proto){
        LoadGraph(model_proto);
    }


    void FBOSession::LoadGraph(const ::dlxnet::ModelProto& model_proto){
        const uint64 start_time_usecs = env_->NowMicros();
        *model_=model_proto;
        // clear kernels first
        kernels_.clear();

        // LOG(INFO)<<"Write proto to Text";
        // WriteProtoToText("./demo.pbtxt", *model_);

        dlxnet::GraphProto graph = model_->graph();
        // create kernel and setup input and output for each node
        // Note that dont need to allocate memory due to lack of shape information
        total_tensors_.resize(graph.tensor_names_size());


        // build tensor_name -> tensor_index map
        for(int i=0;i<graph.tensor_names_size();++i){
            tensor_name_index_[graph.tensor_names(i)] = i;
        }


        Kernel* kernel;
        std::unique_ptr<Kernel> kernel_ptr;
        for(auto& node: graph.node()){
            kernel=nullptr;
            // TODO(breakpoint) handle with input node, ignore it for now
            if(node.type()=="Input"){
                LOG(INFO)<<"Ignore Node Type: "<<node.type();
                continue;
            }
            KernelRegistry::Global()->CreateKernel(node.type(), &kernel, context_);
            kernel_ptr.reset(kernel);
            if(kernel==nullptr){
                LOG(FATAL)<<"unsupported kernel name "<<node.type();
            }
            kernel->set_kernel_name(node.name());
            kernel->set_kernel_type(node.type());
            kernel->set_session(this);

            kernel->SetupAttr(node.attr());

            // setup program for each kernel here
            kernel->SetupProgram(context_->CreateProgram(kernel->kernel_fname()));
            // fill inputs and outputs
            for(int i=0; i<node.input_index_size(); ++i){
                kernel->input_tensor_indexes_.emplace_back(node.input_index(i));
            }
            for(int i=0; i<node.output_index_size(); ++i){
                kernel->output_tensor_indexes_.emplace_back(node.output_index(i));
            }
            kernels_.emplace_back(std::move(kernel_ptr));
        }
        finalized_ = false;
        graph_created_ = true;
        metrics::UpdateGraphBuildTime(env_->NowMicros() - start_time_usecs);
    }



    void FBOSession::Run(const NamedTensorList& inputs_cpu){
        StepStats step_stats;
        Run(inputs_cpu, &step_stats);
    }

    void FBOSession::Run(const NamedTensorList& inputs_cpu,
            StepStats* step_stats){
        std::unique_ptr<StepStatsCollector> step_collector=nullptr;
        if(tracking_stats){
            step_collector.reset(new StepStatsCollector(step_stats));
        }
        // session set up
        {
            uint64 start_time_usecs;
            if(step_collector){
                OPENGL_CALL(glFinish());
                start_time_usecs = env_->NowMicros();
            }
            // Start();
            Setup(inputs_cpu, nullptr);
            // Stop("Setup Time");
            if(step_collector){
                OPENGL_CALL(glFinish());
                float session_setup_time = (env_->NowMicros() - start_time_usecs);
                step_stats->set_all_setup_time_micros(session_setup_time);
            }
        }

        CHECK(finalized_)<<"Please Setup Session First";
        const uint64 start_time_usecs = env_->NowMicros();
        NodeExecStatsInterface* stats = nullptr;

        Start();
        for(int i=0;i<kernels_.size();++i){
            auto kernel = kernels_[i].get();
            if(CheckKernelReady(kernel)){
                continue;
            }
            // Start();
            if(step_collector){
                stats = step_collector->CreateNodeExecStats(kernel);
                auto scheduled_nsec = nodestats::NowInNsec();
                nodestats::SetScheduled(stats, scheduled_nsec);
                nodestats::SetAllStart(stats);
            }

            // op computation time
            // nodestats::SetOpStart(stats);
            kernel->Compute();


            // Stop(kernel->kernel_name()+" "+kernel->kernel_type());
            // for(int i=0;i<kernel->output_tensors_.size();++i){
            // nodestats::SetOutput(stats, i, kernel->output_tensors_[i]);
            // }
            // nodestats::SetOpEnd(stats);

            // // node end time
            if(step_collector){
                nodestats::SetAllEnd(stats);
                stats->Done(kOpenGLDeviceName);
            }

            // // save to collector with device name
        }

        if(step_collector){
            // save data to proto
            step_collector->Finalize();
        }
        Stop("ExecTime");
        metrics::UpdateGraphExecTime(env_->NowMicros() - start_time_usecs);
    }


    FBOSession::FBOSession(Context* context)
        :context_(context){
            // create vertex shader first
            model_ = new dlxnet::ModelProto;
            env_ = Env::Default();
            // reset context for current session
            context_->Reset();
        }

    void FBOSession::AllocateTensor(const TensorShapeList& shapes, TensorList& tensors){
        // tensors.resize(shapes.size());
        tensors.clear();
        for(auto& shape: shapes){
            // only allocate texture tensor in session
            // due to that only texture 2d tensor is used to feed or output
            tensors.emplace_back(new Tensor(Tensor::DT_FLOAT, shape,
                        Tensor::DEVICE_TEXTURE));
        }
    }

    void FBOSession::Setup(const NamedTensorList& inputs_cpu,
            StepStatsCollector* step_collector){
        CHECK(graph_created_)<<"No Graph Loaded!";
        // allocate memory for each tensor
        // so that dont need to allocate input and output tensors
        // for each kernel during computation

        // allocate memory for input tensor(device_tensor) first
        // TODO(breakpoint) add input-typed kernel
        for(auto input_iter=inputs_cpu.begin(); input_iter!=inputs_cpu.end(); ++input_iter){
            const Tensor* input_cpu = input_iter->second;
            const auto& tensor_name = input_iter->first;

            auto iter = tensor_name_index_.find(tensor_name);
            if(iter==tensor_name_index_.end()){
                LOG(FATAL)<<"tensor_name: "<<tensor_name<<" Cannot Find";
            }
            const int input_index = iter->second;

            if(!finalized_){
                // allocate memory in the first time
                total_tensors_[input_index].reset(
                        new Tensor(Tensor::DT_FLOAT, input_cpu->shape(),
                            Tensor::DEVICE_TEXTURE, FormatToStride4(input_cpu->dformat())));
            }
            // upload data, initialize input tensor
            context_->CopyCPUTensorToDevice(input_cpu, total_tensors_[input_index].get());
        }
        if(finalized_){return;}

        ready_.clear();
        ready_.resize(total_tensors_.size());
        std::fill(ready_.begin(), ready_.end(), false);

        for(int i=0;i<kernels_.size();++i){
            auto& kernel = kernels_[i];
            // clear input and output tensors
            kernel->input_tensors_.clear();
            kernel->output_tensors_.clear();
            LOG(INFO)<<"name: " << kernel->kernel_name()
                <<" type: "<<kernel->kernel_type();
            TensorShapeList output_shapes;
            for(int j=0; j<kernel->input_tensor_indexes_.size(); ++j){
                Tensor* input_tensor = total_tensors_[kernel->input_tensor_indexes_[j]].get();
                CHECK(input_tensor)<<"input tensor is uninitialized of kernel index: "<<i;
                kernel->input_tensors_.emplace_back(input_tensor);
            }

            // infer output shapes from input shapes
            // Note that use input tensor as arg instead of input shapes
            // we need dformat(like nhwc) info to derminate the output shape no only the input shape.
            // kernel->InferOutputShape(input_tensors, output_shapes);
            kernel->InferOutputShape(kernel->input_tensors_, output_shapes);
            CHECK_GT(output_shapes.size(), 0);

            // allocate memory for each output tensors according to their shapes
            for(int j=0;j<output_shapes.size();++j){
                auto dformat = kernel->GetOutputDFormat(j);
                auto output_tensor = new Tensor(Tensor::DT_FLOAT, output_shapes[j],
                        Tensor::DEVICE_TEXTURE, dformat);
                total_tensors_[kernel->output_tensor_indexes_[j]].reset(output_tensor);

                kernel->output_tensors_.emplace_back(output_tensor);
            }

            if(CheckKernelReady(kernel.get())){
                // precompute kernel, not only used for constant kernel
                kernel->Compute();
                OPENGL_CHECK_ERROR;
                UpdateKernelReady(kernel.get());
            }

            // log kernel info after kernel finalized
            DLOG(INFO)<<kernel->DebugString();
        }
        finalized_ = true;
        OPENGL_CHECK_ERROR;
    }

    bool FBOSession::CheckKernelReady(const Kernel* kernel){
        // if some kernel is so special, e.g, shape kernel,
        // force it to execute
        if(kernel->ForceReady()){
            return true;
        }
        // find all precondition
        for(auto tensor_id: kernel->input_tensor_indexes_){
            if(not ready_[tensor_id]){
                return false;
            }
        }
        return true;
    }


    void FBOSession::UpdateKernelReady(const Kernel* kernel){
        for(auto tensor_id: kernel->output_tensor_indexes_){
            ready_[tensor_id]=true;
        }
    }

    void FBOSession::GetOutputs(const TensorNameList& output_names,
            const StringList& output_dformats, TensorList* outputs){
        CHECK_EQ(output_names.size(), output_dformats.size());
        const int num_outputs = output_names.size();
        // outputs->clear();
        // outputs->reserve(output_names.size());
        if(outputs->size()<num_outputs){
            outputs->resize(num_outputs);
        }

        int index = 0;
        for(int i=0;i<output_names.size();++i){
            auto tensor_name = output_names[i];
            auto gpu_tensor = FindTensorByName(tensor_name);
            auto dformat_str = output_dformats[index++];
            DataFormat dformat = StrToFormat(dformat_str);
            if(outputs->at(i)==nullptr){
                outputs->at(i) = new Tensor(Tensor::DT_FLOAT, gpu_tensor->shape(),
                        Tensor::HOST_MEMORY, dformat);
            }else{
                // check its shape, it should be same with gpu tensor
            }
            context_->CopyDeviceTensorToCPU(gpu_tensor, outputs->at(i));
        }
    }

    std::string FBOSession::DebugString()const{
        std::string ret_str;
        ret_str+="ModelProto: ";
        ret_str+=model_->DebugString();
        ret_str+="\n";
        return ret_str;
    }

    Tensor* FBOSession::FindTensorByName(const std::string& tensor_name){
        auto iter = tensor_name_index_.find(tensor_name);
        if(iter==tensor_name_index_.end()){
            LOG(FATAL)<<"tensor_name: "<<tensor_name<<" Cannot Find";
        }

        const int tensor_index = tensor_name_index_[tensor_name];
        return FindTensorById(tensor_index);
    }

    Tensor* FBOSession::FindTensorById(const int id){
        CHECK_LT(id, total_tensors_.size());
        return total_tensors_[id].get();
    }
}//namespace opengl
