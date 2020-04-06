#ifndef CORE_GRAPH_EXECUTOR_H_
#define CORE_GRAPH_EXECUTOR_H_
#include <functional>

#include "core/logging.h"
#include "core/status.h"
#include "node_def.pb.h"

class Device;
class OpKernel;


class Executor{
    public:
        ~Executor();
        // args used to running
        struct Args{
        };

        typedef std::function<void(const Status&)> DoneCallback;
        virtual void RunAsync(const Args& args, DoneCallback done)=0;

        virtual Status Run(const Args& args){
            Status ret;
            RunAsync(args, [&ret](const Status& s){
                    ret=s;
                    });
            return ret;
        }
};

// used to construct executor
struct LocalExecutorParams{
    Device* device;
    std::function<Status(const NodeDef&, OpKernel**)> create_kernel;
    std::function<void(OpKernel*)> delete_kernel;
};


#endif
