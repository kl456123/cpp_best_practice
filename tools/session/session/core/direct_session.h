#ifndef SESSION_CORE_DIRECT_SESSION_H_
#define SESSION_CORE_DIRECT_SESSION_H_
#include <vector>
#include <atomic>

#include "session/core/session_factory.h"
#include "session/core/session_options.h"
#include "session/core/session_state.h"
#include "session/core/session.h"
#include "session/core/tensor.h"

#include "session/core/device_mgr.h"
#include "session/core/device.h"
#include "session/core/device_set.h"
#include "session/core/graph_execution_state.h"

#include "session/core/threadpool.h"

#include "session/utils/macros.h"
#include "session/core/graph.h"
#include "session/core/build_graph_options.h"
#include "session/core/costmodel.h"
#include "session/core/costmodel_manager.h"
#include "session/core/threadpool_options.h"
#include "session/core/function.h"
#include "session/core/executor.h"

class DeviceMgr;
class DirectSessionFactory;
class Node;
class GraphExecutionState;

typedef std::vector<std::pair<std::string, Tensor>> NamedTensorList;

class DirectSession: public Session{
    public:
        DirectSession(const SessionOptions& options, const DeviceMgr* device_mgr,
                DirectSessionFactory* const factory);
        ~DirectSession()override;
        typedef std::vector<std::pair<std::string, Tensor>> NamedTensorList;
        typedef std::unordered_map<std::string, Node*> NameNodeMap;
        Status Create(const GraphDef& graph) override;
        Status Create(GraphDef&& graph) override;
        Status Extend(const GraphDef& graph) override;
        Status Extend(GraphDef&& graph) override;
        Status ExtendLocked(GraphDef graph);
        Status CheckNotClosed() {
            if (closed_) return errors::Cancelled("Session has been closed.");
            return Status::OK();
        }

        Status Run(const NamedTensorList& inputs,
                const std::vector<string>& output_names,
                const std::vector<string>& target_nodes,
                std::vector<Tensor>* outputs) override;

        // NOTE: Experimental and subject to change.
        Status Run(const RunOptions& run_options,
                const NamedTensorList& inputs,
                const std::vector<string>& output_names,
                const std::vector<string>& target_nodes,
                std::vector<Tensor>* outputs,
                RunMetadata* run_metadata) override;

        // NOTE: Experimental and subject to change.
        Status Run(const RunOptions& run_options,
                const NamedTensorList& inputs, const std::vector<string>& output_names,
                const std::vector<string>& target_nodes, std::vector<Tensor>* outputs,
                RunMetadata* run_metadata, const thread::ThreadPoolOptions& threadpool_options) override;
        Status ListDevices(
                std::vector<DeviceAttributes>* response) override;
        Status Close() override;
        Status LocalDeviceManager(const DeviceMgr** output) override {
            *output = device_mgr_.get();
            return Status::OK();
        }

        Status CheckGraphCreated(const char* method) {
            if (!graph_created_) {
                return errors::InvalidArgument(
                        "Session was not created with a graph before ", method, "!");
            }
            return Status::OK();
        }



        void ExportCostModels(CostModelManager::CostModelMap* cost_models) {
            cost_model_manager_.ExportCostModels(cost_models);
        }
        Status Finalize() override;

        const SessionOptions& options() const { return options_; }
    private:
        struct PerPartitionExecutorsAndLib{
            std::unique_ptr<Graph> graph=nullptr;
            Device* device=nullptr;
            std::unique_ptr<Executor> executor;
        };
        const SessionOptions options_;

        // device
        const std::unique_ptr<const DeviceMgr> device_mgr_;
        std::vector<Device*> devices_;
        DeviceSet device_set_;

        // session id
        std::string session_handle_;
        bool graph_created_=false;
        bool finalized_=false;

        std::vector<std::pair<thread::ThreadPool*, bool>> thread_pools_;
        Status init_error_;
        bool sync_on_finish_=true;
        // An ExecutorsAndKeys is created for a given set of feeds/fetches.
        // 'step_count' is the number of times this graph is executed.
        // 'graph' is the entire graph being executed. 'name_to_node'
        // maps node name to node. We keep 'graph' and 'name_to_node' only in
        // the case of partial runs. Each item in 'items' is the executor for
        // a partition of the graph bundled with its dependent library runtime.
        // 'input_keys' are the rendezvous keys for the feeds and 'output_keys'
        // are rendezvous keys for the fetches.
        struct ExecutorsAndKeys {
            ExecutorsAndKeys() : step_count(0) {}

            std::atomic_int_fast64_t step_count;
            std::unique_ptr<Graph> graph;
            NameNodeMap name_to_node;
            std::vector<PerPartitionExecutorsAndLib> items;
            std::unordered_map<std::string, size_t> input_name_to_index;
            std::unordered_map<std::string, std::string> input_name_to_rendezvous_key;
            std::unordered_map<std::string, size_t> output_name_to_index;
            std::unordered_map<std::string, std::string> output_name_to_rendezvous_key;

            std::vector<DataType> input_types;
            std::vector<DataType> output_types;
            CallableOptions callable_options;

            int64_t collective_graph_key = BuildGraphOptions::kNoCollectiveGraphKey;
        };

        struct RunStateArgs{
            bool is_partical_run = false;
            std::string handle;
            std::unique_ptr<Graph> graph;
            int64_t collective_graph_key = BuildGraphOptions::kNoCollectiveGraphKey;
        };

        struct RunState{
            Status status;
            RunState(int64_t step_id, const std::vector<Device*>* devices);
        };

        // Retrieves an already existing set of executors to run 'inputs' and
        // 'outputs', or creates and caches them for future use.
        Status GetOrCreateExecutors(
                std::vector<std::string> inputs, std::vector<std::string> outputs,
                std::vector<std::string> target_nodes,
                ExecutorsAndKeys** executors_and_keys, RunStateArgs* run_state_args);

        // Creates a set of executors to run the subgraph defined by
        // `callable_options`.
        Status CreateExecutors(const CallableOptions& callable_options,
                std::unique_ptr<ExecutorsAndKeys>* out_executors_and_keys,
                RunStateArgs* run_state_args);

        // // Creates several graphs given the existing graph_def_ and the
        // // input feeds and fetches, given 'devices'. The graphs share a common
        // // function library 'flib_def'.
        Status CreateGraphs(const BuildGraphOptions& options,
                std::unordered_map<std::string, std::unique_ptr<Graph>>* outputs,
                RunStateArgs* run_state_args, std::vector<DataType>* input_types,
                std::vector<DataType>* output_types, int64_t* collective_graph_key);

        Status RunInternal(
                int64_t step_id, const RunOptions& run_options,
                CallFrameInterface* call_frame, ExecutorsAndKeys* executors_and_keys,
                RunMetadata* run_metadata,
                const thread::ThreadPoolOptions& threadpool_options);

        // executors
        std::unordered_map<std::string, std::shared_ptr<ExecutorsAndKeys>> executors_;
        std::unordered_map<std::string, std::string> stateful_placements_;
        std::unique_ptr<GraphExecutionState> execution_state_;
        bool closed_ = false;

        SessionState session_state_;
        DirectSessionFactory* const factory_;

        std::atomic<int64_t> edge_name_counter_ = {0};
        std::atomic<int64_t> handle_name_counter_ = {0};
        static std::atomic_int_fast64_t step_id_counter_;
        CostModelManager cost_model_manager_;

        bool run_in_caller_thread_ = false;
        DISALLOW_COPY_AND_ASSIGN(DirectSession);
};







#endif
