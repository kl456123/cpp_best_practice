#include "session/core/executor.h"
#include "session/core/executor_factory.h"

class ExecutorImpl :public Executor{
    public:
        explicit ExecutorImpl(const LocalExecutorParams& p):params_(p){
            // CHECK(p.create_kernel!=nullptr);
            // CHECK(p.delete_kernel!=nullptr);
        }
        ~ExecutorImpl()override{
        }
        Status Initialize(const Graph& graph);
        void RunAsync(const Args& args, DoneCallback done) override;
    private:
        // Owned.
        LocalExecutorParams params_;
        DISALLOW_COPY_AND_ASSIGN(ExecutorImpl);

};
void ExecutorImpl::RunAsync(const Args& args, DoneCallback done) {
    // (new ExecutorState(args, this))->RunAsync(std::move(done));
    LOG(INFO)<<"ExecutorImpl is RunAsyncing";
}

Status ExecutorImpl::Initialize(const Graph& graph){
}

Status NewLocalExecutor(const LocalExecutorParams& params, const Graph& graph,
        Executor** executor) {
    ExecutorImpl* impl = new ExecutorImpl(params);
    const Status s = impl->Initialize(graph);
    if (s.ok()) {
        *executor = impl;
    } else {
        delete impl;
    }
    return s;
}

namespace{
    class DefaultExecutorRegistrar{
        public:
            DefaultExecutorRegistrar(){
                Factory* factory = new Factory;
                ExecutorFactory::Register("", factory);
                ExecutorFactory::Register("DEFAULT", factory);
            }
        private:
            class Factory:public ExecutorFactory{
                Status NewExecutor(const LocalExecutorParams& params, const Graph& graph,
                        std::unique_ptr<Executor>* out_executor)override{
                    Executor* ret = nullptr;
                    RETURN_IF_ERROR(NewLocalExecutor(params, std::move(graph), &ret));
                    out_executor->reset(ret);
                    return Status::OK();
                }
            };
    };
}
