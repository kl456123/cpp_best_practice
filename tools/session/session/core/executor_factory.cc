#include "session/core/executor_factory.h"
#include "session/utils/errors.h"
#include "session/utils/status.h"

namespace {

typedef std::unordered_map<string, ExecutorFactory*> ExecutorFactories;
ExecutorFactories* executor_factories() {
  static ExecutorFactories* factories = new ExecutorFactories;
  return factories;
}

}  // namespace

void ExecutorFactory::Register(const string& executor_type,
                               ExecutorFactory* factory) {
  if (!executor_factories()->insert({executor_type, factory}).second) {
    LOG(FATAL) << "Two executor factories are being registered "
               << "under" << executor_type;
  }
}

namespace {
const string RegisteredFactoriesErrorMessageLocked(){
  std::vector<string> factory_types;
  for (const auto& executor_factory : *executor_factories()) {
    factory_types.push_back(executor_factory.first);
  }
  return string_utils::str_cat("Registered factories are {",
                         factory_types[0], "}.");
}
}  // namespace

Status ExecutorFactory::GetFactory(const string& executor_type,
                                   ExecutorFactory** out_factory) {
  auto iter = executor_factories()->find(executor_type);
  if (iter == executor_factories()->end()) {
    return errors::NotFound(
        "No executor factory registered for the given executor type: ",
        executor_type, " ", RegisteredFactoriesErrorMessageLocked());
  }

  *out_factory = iter->second;
  return Status::OK();
}

Status NewExecutor(const string& executor_type,
                   const LocalExecutorParams& params, const Graph& graph,
                   std::unique_ptr<Executor>* out_executor) {
  ExecutorFactory* factory = nullptr;
  RETURN_IF_ERROR(ExecutorFactory::GetFactory(executor_type, &factory));
  return factory->NewExecutor(params, std::move(graph), out_executor);
}

