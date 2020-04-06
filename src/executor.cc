#include <vector>

#include "core/graph/executor.h"



class ExecutorImpl: public Executor{
    public:
        explicit ExecutorImpl(const LocalExecutorParams& param):param_(param){
        }
        virtual void RunAsync(const Args& args)override;
        Status Initialize(const Graph& graph);

    private:
        LocalExecutorParams param_;
        std::vector<const Node*> root_nodes_;
};

class ExecutorState{
    Status Process();
    void RunAsync(const Args& args);
    Status PrepareInputs();
    Status ProcessOutputs();
    void Finish();
};

Status ExecutorImpl::RunAsync(const Args& args, DoneCallback done){
    (new ExecutorState(args, this))->RunAsync(done);
}

Status ExecutorImpl::Initialize(const Graph& graph) {
    // preprocess ops
    for(auto& node: param_.nodes){
        OpKernel* op_kernel=nullptr;
        Status s = param_.create_kernel(node.node_def_, &op_kernel);
        if(!s.ok()){
            LOG(ERROR)<<"Executor failed to create kernel. "<<s;
            return s;
        }
    }
}

void ExecutorImpl::RunAsync(DoneCallback done){
    Status s = Process();
    done(s);
}

Status ExecutorState::Process(){
    for(const Node* item: impl_->root_nodes_){
        DCHECK_EQ(item->num_inputs, 0);
        ready.push_back();
    }
    Device* device = impl_->params_.device;

    // bfs
    while(!ready.empty()){
        node = ready.front();

        // construct OpKernelContext
        OpKernelContext ctx(&params);
        OpKernel* op_kernel = node.op_kernel_;

        device->Compute(op_kernel, &ctx);
    }
}




Status ExecutorState::PrepareInputs(){}

Status ExecutorState::ProcessOutputs(){}







