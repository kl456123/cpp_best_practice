#include "session/core/direct_session.h"
#include "session/utils/status.h"
#include "session/utils/strcat.h"
#include "session/utils/errors.h"
#include "session/utils/random.h"
#include "session/core/device_factory.h"
#include "session/core/executor_factory.h"
#include "session/core/graph_partition.h"
#include <memory>

namespace{
}

class DirectSessionFactory : public SessionFactory{
    public:
        DirectSessionFactory(){}

        Status NewSession(const SessionOptions& options,
                Session** out_session)override{

            std::vector<std::unique_ptr<Device>> devices;
            RETURN_IF_ERROR(DeviceFactory::AddDevices(options, "/job:localhost/replica:0/task:0", &devices));
            DirectSession* session = new DirectSession(options, new StaticDeviceMgr(std::move(devices)), this);
            *out_session = session;
            sessions_.push_back(session);
            return Status::OK();
        }

        bool AcceptsOptions(const SessionOptions& options)override{
            return options.target.empty();
        }

        void Deregister(const DirectSession* session){
            sessions_.erase(std::remove(sessions_.begin(),
                        sessions_.end(), session), sessions_.end());
        }
    private:
        std::vector<DirectSession*> sessions_;

};

// auto register
class DirectSessionRegistrar{
    public:
        DirectSessionRegistrar(){
            SessionFactory::Register("DIRECT_SESSION", new DirectSessionFactory);
        }
};

static DirectSessionRegistrar registrar;
std::atomic_int_fast64_t DirectSession::step_id_counter_(1);

DirectSession::DirectSession(const SessionOptions& options,
        const DeviceMgr* device_mgr,
        DirectSessionFactory* const factory)
    :options_(options), device_mgr_(device_mgr), factory_(factory){
        // thread pools
        const int thread_pool_size = options_.config.session_inter_op_thread_pool_size();
        if(thread_pool_size>0){
            for(int i=0;i<thread_pool_size;i++){
            }
        }else{
            thread_pools_.emplace_back();
        }

        session_handle_= string_utils::str_cat("direct", std::to_string(random_utils::New64()));

        // device
        if(options.config.log_device_placement()){
        }
        for(auto d:device_mgr_->ListDevices()){
            devices_.push_back(d);
            device_set_.AddDevice(d);
            d->op_segment()->AddHold(session_handle_);
        }
    }

DirectSession::~DirectSession(){
    if(!closed_){
        Close().IgnoreError();
    }
    for(auto d:device_mgr_->ListDevices()){
        d->op_segment()->RemoveHold(session_handle_);
    }
    execution_state_.reset();
}

Status DirectSession::Close() {
    // cancellation_manager_->StartCancel();
    {
        // mutex_lock l(closed_lock_);
        if (closed_) return Status::OK();
        closed_ = true;
    }
    if (factory_ != nullptr) factory_->Deregister(this);
    return Status::OK();
}

Status DirectSession::Create(const GraphDef& graph){
    return Create(GraphDef(graph));
}

Status DirectSession::Create(GraphDef&& graph){
    RETURN_IF_ERROR(init_error_);
    if(graph.node_size()>0){
        if(graph_created_){
            return errors::AlreadyExists("A Graph has already been created for this session.");
        }
        return ExtendLocked(std::move(graph));
    }
    return Status::OK();
}

Status DirectSession::Extend(const GraphDef& graph) {
    return Extend(GraphDef(graph));
}

Status DirectSession::Extend(GraphDef&& graph) {
    RETURN_IF_ERROR(CheckNotClosed());
    return ExtendLocked(std::move(graph));
}

Status DirectSession::ExtendLocked(GraphDef graph) {
    if (finalized_) {
        return errors::FailedPrecondition("Session has been finalized.");
    }
    if (!execution_state_) {
        // If this is the first call, we can initialize the execution state
        // with `graph` and do not need to call `Extend()`.
        // NOTE(mrry): The function library created here will be used for
        // all subsequent extensions of the graph.
        // flib_def_.reset(
        // new FunctionLibraryDefinition(OpRegistry::Global(), graph.library()));
        GraphExecutionStateOptions options;
        options.device_set = &device_set_;
        options.session_options = &options_;
        options.session_handle = session_handle_;
        RETURN_IF_ERROR(GraphExecutionState::MakeForBaseGraph(
                    std::move(graph), options, &execution_state_));
        graph_created_ = true;
    } else {
        // TF_RETURN_IF_ERROR(flib_def_->AddLibrary(graph.library()));
        std::unique_ptr<GraphExecutionState> state;
        // TODO(mrry): Rewrite GraphExecutionState::Extend() to take `graph` by
        // value and move `graph` in here.
        RETURN_IF_ERROR(execution_state_->Extend(graph, &state));
        execution_state_.swap(state);
    }
    return Status::OK();
}


Status DirectSession::Finalize() {
    if (finalized_) {
        return errors::FailedPrecondition("Session already finalized.");
    }
    if (!graph_created_) {
        return errors::FailedPrecondition("Session not yet created.");
    }
    execution_state_.reset();
    finalized_ = true;
    return Status::OK();
}

Status DirectSession::ListDevices(
        std::vector<DeviceAttributes>* response) {
    response->clear();
    response->reserve(devices_.size());
    for (Device* d : devices_) {
        const DeviceAttributes& attrs = d->attributes();
        response->emplace_back(attrs);
    }
    return Status::OK();
}

Status DirectSession::Run(const RunOptions& run_options,
        const NamedTensorList& inputs, const std::vector<string>& output_names,
        const std::vector<string>& target_nodes, std::vector<Tensor>* outputs,
        RunMetadata* run_metadata, const thread::ThreadPoolOptions& threadpool_options){
    RETURN_IF_ERROR(CheckNotClosed());
    RETURN_IF_ERROR(CheckGraphCreated("Run()"));
    // Extract the inputs names for this run of the session.
    std::vector<string> input_tensor_names;
    input_tensor_names.reserve(inputs.size());
    size_t input_size = 0;
    for (const auto& it : inputs) {
        input_tensor_names.push_back(it.first);
        input_size += it.second.AllocatedBytes();
    }
    ExecutorsAndKeys* executors_and_keys=nullptr;
    RunStateArgs run_state_args;
    RETURN_IF_ERROR(GetOrCreateExecutors(input_tensor_names, output_names,
                target_nodes, &executors_and_keys, &run_state_args));

    // Configure a call frame for the step, which we use to feed and
    // fetch values to and from the executors.
    FunctionCallFrame call_frame(executors_and_keys->input_types,
            executors_and_keys->output_types);
    std::vector<Tensor> feed_args(inputs.size());
    for(const auto& it: inputs){
        if(false){
        }
    }
    const Status s = call_frame.SetArgs(feed_args);
    if(errors::IsInternal(s)){
        return errors::InvalidArgument(s.error_message());
    }else if(!s.ok()){
        return s;
    }
    const int64 step_id = step_id_counter_.fetch_add(1);
    RETURN_IF_ERROR(RunInternal(step_id, run_options, &call_frame,
                executors_and_keys, run_metadata, threadpool_options));
    if(outputs){
    }
    return Status::OK();
}


Status DirectSession::Run(const RunOptions& run_options,
        const NamedTensorList& inputs,
        const std::vector<string>& output_names,
        const std::vector<string>& target_nodes,
        std::vector<Tensor>* outputs,
        RunMetadata* run_metadata) {
    return Run(run_options, inputs, output_names, target_nodes, outputs,
            run_metadata, thread::ThreadPoolOptions());
}

Status DirectSession::Run(const NamedTensorList& inputs,
        const std::vector<string>& output_names,
        const std::vector<string>& target_nodes,
        std::vector<Tensor>* outputs) {
    RunMetadata run_metadata;
    return Run(RunOptions(), inputs, output_names, target_nodes, outputs,
            &run_metadata);
}

Status DirectSession::GetOrCreateExecutors(
        std::vector<std::string> inputs, std::vector<std::string> outputs,
        std::vector<std::string> target_nodes,
        ExecutorsAndKeys** executors_and_keys, RunStateArgs* run_state_args){
    std::string key = "";
    for(int i=0;i<inputs.size();i++){
        if(i>0) key+=",";
        key+= std::string(inputs[i]);
    }
    key+="->";

    for(int i=0;i<outputs.size();i++){
        if(i>0) key+=",";
        key+= std::string(outputs[i]);
    }
    auto it = executors_.find(key);
    if (it != executors_.end()) {
        *executors_and_keys = it->second.get();
        return Status::OK();
    }

    // Nothing found, so create the executors and store in the cache.
    // The executor_lock_ is intentionally released while executors are
    // being created.
    CallableOptions callable_options;
    callable_options.mutable_feed()->Reserve(inputs.size());
    for (const string& input : inputs) {
        callable_options.add_feed(input);
    }
    callable_options.mutable_fetch()->Reserve(outputs.size());
    for (const string& output : outputs) {
        callable_options.add_fetch(output);
    }
    callable_options.mutable_target()->Reserve(target_nodes.size());
    for (const string& target : target_nodes) {
        callable_options.add_target(target);
    }
    // *callable_options.mutable_run_options()->mutable_debug_options() =
    // run_state_args->debug_options;
    // callable_options.mutable_run_options()
    // ->mutable_experimental()
    // ->set_collective_graph_key(run_state_args->collective_graph_key);
    std::unique_ptr<ExecutorsAndKeys> ek;
    // std::unique_ptr<FunctionInfo> func_info;
    RETURN_IF_ERROR(CreateExecutors(callable_options, &ek, run_state_args));

    *executors_and_keys = ek.get();
    executors_.emplace(key, std::shared_ptr<ExecutorsAndKeys>(std::move(ek)));

    return Status::OK();
}

Status DirectSession::CreateExecutors(
        const CallableOptions& callable_options,
        std::unique_ptr<ExecutorsAndKeys>* out_executors_and_keys,
        RunStateArgs* run_state_args){
    BuildGraphOptions options;
    options.callable_options = callable_options;

    // build graphs
    std::unique_ptr<ExecutorsAndKeys> ek(new ExecutorsAndKeys);
    ek->callable_options = callable_options;
    std::unordered_map<string, std::unique_ptr<Graph>> graphs;
    RETURN_IF_ERROR(CreateGraphs(
                options, &graphs, run_state_args, &ek->input_types,
                &ek->output_types, &ek->collective_graph_key));
    ek->items.reserve(graphs.size());
    const SessionMetadata* session_metadata = nullptr;
    for(auto iter=graphs.begin(); iter!=graphs.end(); ++iter){
        const string& partition_name = iter->first;
        std::unique_ptr<Graph>& partition_graph = iter->second;
        Device* device;
        RETURN_IF_ERROR(device_mgr_->LookupDevice(partition_name, &device));
        ek->items.resize(ek->items.size() + 1);
        auto* item = &(ek->items.back());
        LocalExecutorParams params;
        params.device = device;
        params.session_metadata = session_metadata;
        // params.function_library = lib;
        auto opseg = device->op_segment();
        // params.create_kernel = [this, lib, opseg](const NodeDef& ndef,
        // OpKernel** kernel) {
        // // NOTE(mrry): We must not share function kernels (implemented
        // // using `CallOp`) between subgraphs, because `CallOp::handle_`
        // // is tied to a particular subgraph. Even if the function itself
        // // is stateful, the `CallOp` that invokes it is not.
        // if (!OpSegment::ShouldOwnKernel(lib, ndef.op())) {
        // return lib->CreateKernel(ndef, kernel);
        // }
        // auto create_fn = [lib, &ndef](OpKernel** kernel) {
        // return lib->CreateKernel(ndef, kernel);
        // };
        // // Kernels created for subgraph nodes need to be cached.  On
        // // cache miss, create_fn() is invoked to create a kernel based
        // // on the function library here + global op registry.
        // return opseg->FindOrCreate(session_handle_, ndef.name(), kernel,
        // create_fn);
        // };
        // params.delete_kernel = [lib](OpKernel* kernel) {
        // if (kernel && !OpSegment::ShouldOwnKernel(lib, kernel->type_string()))
        // delete kernel;
        // };
        item->executor = nullptr;
        item->device = device;
        const std::string executor_type = "";
        RETURN_IF_ERROR(NewExecutor(executor_type, params, *partition_graph, &item->executor));
        item->graph = std::move(partition_graph);
    }
    *out_executors_and_keys = std::move(ek);

    return Status::OK();

}

Status DirectSession::RunInternal(int64_t step_id, const RunOptions& run_options,
        CallFrameInterface* call_frame, ExecutorsAndKeys* executors_and_keys,
        RunMetadata* run_metadata,
        const thread::ThreadPoolOptions& threadpool_options){
    const uint64_t start_time_usecs = options_.env->NowMicros();
    const int64_t executor_step_count = executors_and_keys->step_count.fetch_add(1);
    RunState run_state(step_id, &devices_);
    if(executors_and_keys->collective_graph_key!=BuildGraphOptions::kNoCollectiveGraphKey){
    }

    // thread pool
    // Use std::unique_ptr to ensure garbage collection
    std::unique_ptr<thread::ThreadPool> threadpool_wrapper;
    thread::ThreadPool* pool = nullptr;

    if (run_options.inter_op_thread_pool() < -1 ||
            run_options.inter_op_thread_pool() >=
            static_cast<int32>(thread_pools_.size())) {
        return errors::InvalidArgument("Invalid inter_op_thread_pool: ",
                std::to_string(run_options.inter_op_thread_pool()));
    }
    if (run_in_caller_thread_) {
        pool = nullptr;
    } else if (threadpool_options.inter_op_threadpool != nullptr) {
        threadpool_wrapper = std::make_unique<thread::ThreadPool>(
                threadpool_options.inter_op_threadpool);
        pool = threadpool_wrapper.get();
    } else if (run_options.inter_op_thread_pool() >= 0) {
        pool = thread_pools_[run_options.inter_op_thread_pool()].first;
    }

    if (pool == nullptr) {
        // We allow using the caller thread only when having a single executor
        // specified.
        if (executors_and_keys->items.size() > 1) {
            pool = thread_pools_[0].first;
        } else {
            LOG(INFO) << "Executing Session::Run() synchronously!";
        }
    }
    const bool can_execute_synchronously = pool == nullptr;

    Status run_status;
    auto set_threadpool_args_for_item = [](const PerPartitionExecutorsAndLib& item,
            Executor::Args* args){
    };

    Executor::Args args;
    args.step_id = step_id;
    args.call_frame = call_frame;
    // args.collective_executor =
    // (run_state.collective_executor ? run_state.collective_executor->get()
    // : nullptr);
    args.session_state = &session_state_;
    args.session_handle = session_handle_;
    // args.tensor_store = &run_state.tensor_store;
    // args.step_container = &run_state.step_container;
    args.sync_on_finish = sync_on_finish_;
    args.user_intra_op_threadpool = threadpool_options.intra_op_threadpool;

    bool update_cost_model = false;
    if (can_execute_synchronously) {
        //sync
        const auto& item = executors_and_keys->items[0];
        set_threadpool_args_for_item(item, &args);
        run_status = item.executor->Run(args);
    }else{
        //async
        ExecutorBarrier* barrier = new ExecutorBarrier();
        for (const auto& item : executors_and_keys->items) {
            set_threadpool_args_for_item(item, &args);
            item.executor->RunAsync(args, barrier->Get());
        }
        run_status = run_state.status;
    }
    RETURN_IF_ERROR(run_status);

    if(update_cost_model){
    }

    return Status::OK();
}
DirectSession::RunState::RunState(int64_t step_id,
        const std::vector<Device*>* devices){}

Status DirectSession::CreateGraphs(const BuildGraphOptions& options,
        std::unordered_map<std::string, std::unique_ptr<Graph>>* outputs,
        RunStateArgs* run_state_args, std::vector<DataType>* input_types,
        std::vector<DataType>* output_types, int64_t* collective_graph_key){
    if(finalized_){
        return errors::FailedPrecondition("Session has been finalized.");
    }
    std::unique_ptr<ClientGraph> client_graph;

    return Status::OK();
}
