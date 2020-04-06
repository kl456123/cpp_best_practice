#ifndef CORE_GRAPH_EXECUTOR_FACTORY_H_
#define CORE_GRAPH_EXECUTOR_FACTORY_H_
#include <string>
#include <memory>

#include "core/logging.h"
#include "core/executor.h"
#include "core/graph/graph.h"
#include "stream_executor/platform/status.h"


using std::string;

class ExecutorFactory{
    public:
        virtual Status NewExecutor(const LocalExecutorParams& params,
                const Graph& graph, std::unique_ptr<Executor>* out_executor)=0;

        virtual ~ExecutorFactory() {}

        static void Register(const string& executor_type, ExecutorFactory* factory);
        static Status GetFactory(const string& executor_type,
                ExecutorFactory** out_factory);
};


Status NewExecutor(const string& executor_type,
                   const LocalExecutorParams& params, const Graph& graph,
                   std::unique_ptr<Executor>* out_executor);


#endif
