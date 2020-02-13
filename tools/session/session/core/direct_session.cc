#include "session/core/direct_session.h"
#include "session/utils/status.h"
#include "session/utils/strcat.h"
#include "session/utils/errors.h"
#include "session/core/device_factory.h"

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

        session_handle_= string_utils::str_cat("direct", "1254536346");

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

Status DirectSession::Run(const NamedTensorList& inputs,
        const std::vector<string>& output_names,
        const std::vector<string>& target_nodes,
        std::vector<Tensor>* outputs) {
    // RunMetadata run_metadata;
    // return Run(RunOptions(), inputs, output_names, target_nodes, outputs,
    // &run_metadata);
    return Status::OK();
}
