#ifndef SESSION_CORE_EXECUTOR_H_
#define SESSION_CORE_EXECUTOR_H_
#include <functional>

#include "session/core/device.h"
#include "session/utils/status.h"
#include "session/core/op_kernel.h"
#include "session/core/graph.h"
#include "session/core/threadpool_options.h"
#include "session/core/function.h"
#include "session/core/session_state.h"

#include "node_def.pb.h"
#include "config.pb.h"

class Executor{
    public:
        ~Executor(){}

        struct Args{
            int64_t step_id = 0;
            CallFrameInterface* call_frame = nullptr;
            SessionState* session_state = nullptr;
            bool sync_on_finish=false;
            // Unique session identifier. Can be empty.
            string session_handle;
            thread::ThreadPoolInterface* user_intra_op_threadpool = nullptr;
        };
        typedef std::function<void(const Status&)> DoneCallback;
        virtual void RunAsync(const Args& args, DoneCallback done)=0;

        virtual Status Run(const Args& args){
            Status ret;
            RunAsync(args, [&ret](const Status& s){
                    ret = s;
                    });
            return ret;
        }


};

// Creates an Executor that computes the given "graph".
//
// If successful, returns the constructed executor in "*executor". Otherwise,
// returns an error status.
//
// "params" provides a set of context for the executor. We expect that
// different context would provide different implementations.
struct LocalExecutorParams {
    Device* device;

    const SessionMetadata* session_metadata = nullptr;

    // create_kernel returns an instance of op kernel based on NodeDef.
    // delete_kernel is called for every kernel used by the executor
    // when the executor is deleted.
    std::function<Status(const NodeDef&, OpKernel**)> create_kernel;
    std::function<void(OpKernel*)> delete_kernel;

};
Status NewLocalExecutor(const LocalExecutorParams& params,
        const Graph& graph, Executor** executor);
class ExecutorBarrier{
    public:
        typedef std::function<void(const Status&)> StatusCallback;
        ExecutorBarrier(){}
        ~ExecutorBarrier() {}
        StatusCallback Get() {
            return std::bind(&ExecutorBarrier::WhenDone, this,std::placeholders::_1);;
        }
    private:
        void WhenDone(const Status& s) {
        }
};
#endif
