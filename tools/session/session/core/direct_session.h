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

#include "session/core/thread_pool.h"

#include "session/utils/macros.h"
#include "session/core/graph.h"
#include "session/core/build_graph_options.h"
#include "session/core/costmodel.h"
#include "session/core/costmodel_manager.h"

class DeviceMgr;
class DirectSessionFactory;
class Node;
class GraphExecutionState;

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
        Status ListDevices(
                std::vector<DeviceAttributes>* response) override;
        Status Close() override;
        Status LocalDeviceManager(const DeviceMgr** output) override {
            *output = device_mgr_.get();
            return Status::OK();
        }

        void ExportCostModels(CostModelManager::CostModelMap* cost_models) {
            cost_model_manager_.ExportCostModels(cost_models);
        }
        Status Finalize() override;

        const SessionOptions& options() const { return options_; }
    private:
        const SessionOptions options_;

        // device
        const std::unique_ptr<const DeviceMgr> device_mgr_;
        std::vector<Device*> devices_;
        DeviceSet device_set_;

        // session id
        std::string session_handle_;
        bool graph_created_;
        bool finalized_;

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
            // std::vector<PerPartitionExecutorsAndLib> items;
            std::unordered_map<std::string, size_t> input_name_to_index;
            std::unordered_map<std::string, std::string> input_name_to_rendezvous_key;
            std::unordered_map<std::string, size_t> output_name_to_index;
            std::unordered_map<std::string, std::string> output_name_to_rendezvous_key;

            std::vector<DataType> input_types;
            std::vector<DataType> output_types;

            int64_t collective_graph_key = BuildGraphOptions::kNoCollectiveGraphKey;
        };

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

        DISALLOW_COPY_AND_ASSIGN(DirectSession);
};







#endif
