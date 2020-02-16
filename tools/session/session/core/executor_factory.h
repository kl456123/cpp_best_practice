#ifndef SESSION_CORE_EXECUTOR_FACTORY_H_
#define SESSION_CORE_EXECUTOR_FACTORY_H_
#include <memory>
#include <string>

#include "session/utils/status.h"
#include "session/core/graph.h"
#include "session/core/executor.h"

class Graph;
class Executor;
struct LocalExecutorParams;



class ExecutorFactory {
 public:
  virtual Status NewExecutor(const LocalExecutorParams& params,
                             const Graph& graph,
                             std::unique_ptr<Executor>* out_executor) = 0;
  virtual ~ExecutorFactory() {}

  static void Register(const std::string& executor_type, ExecutorFactory* factory);
  static Status GetFactory(const std::string& executor_type,
                           ExecutorFactory** out_factory);
};

Status NewExecutor(const std::string& executor_type,
                   const LocalExecutorParams& params, const Graph& graph,
                   std::unique_ptr<Executor>* out_executor);
#endif
