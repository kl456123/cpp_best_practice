#ifndef SESSION_CORE_GRAPH_EXECUTION_STATA_H_
#define SESSION_CORE_GRAPH_EXECUTION_STATA_H_
#include <string>
#include <vector>

#include "session/core/graph.h"
#include "session/utils/status.h"
#include "session/core/device_set.h"
#include "session/core/session_options.h"
#include "session/utils/macros.h"
#include "session/core/build_graph_options.h"
#include "session/core/costmodel.h"
#include "session/core/subgraph.h"
#include "session/core/function.h"

#include "graph.pb.h"
namespace subgraph{
    struct RewriteGraphMetadata;
}

using std::string;

struct GraphExecutionStateOptions {
    const DeviceSet* device_set = nullptr;
    const SessionOptions* session_options = nullptr;
    // Unique session identifier. Can be empty.
    string session_handle;
    // A map from node name to device name, representing the unchangeable
    // placement of stateful nodes.
    std::unordered_map<string, string> stateful_placements;
};

struct ClientGraph{
    explicit ClientGraph(
            std::unique_ptr<OpRegistryInterface> flib,
            std::vector<DataType> feed_types, std::vector<DataType> fetch_types,
            int64_t collective_graph_key)
        :graph(flib.get()),
        feed_types(std::move(feed_types)),
        fetch_types(std::move(fetch_types)),
        collective_graph_key(collective_graph_key) {}
    Graph graph;
    std::vector<DataType> feed_types;
    std::vector<DataType> fetch_types;
    int64_t collective_graph_key;
};

class GraphExecutionState{
    public:
        virtual ~GraphExecutionState();
        // Creates a new `GraphExecutionState` for the given
        // `graph_def`, which represents the entire graph for a session.
        static Status MakeForBaseGraph(
                GraphDef&& graph_def, const GraphExecutionStateOptions& options,
                std::unique_ptr<GraphExecutionState>* out_state);
        // Creates a new `GraphExecutionState` and `SimpleClientGraph`
        // for the subgraph of `original_graph_def` defined by
        // `subgraph_options`.
        static Status MakeForPrunedGraph(
                const GraphExecutionState& base_execution_state,
                const GraphExecutionStateOptions& options,
                const BuildGraphOptions& subgraph_options,
                std::unique_ptr<GraphExecutionState>* out_state,
                std::unique_ptr<ClientGraph>* out_client_graph);

        // Creates a new GraphExecutionState representing the
        // concatenation of this graph, and the graph defined by
        // "extension_def". The same name may not be used to define a node
        // in both this graph and "extension_def".
        //
        // If successful, returns OK and the caller takes ownership of "*out".
        // Otherwise returns an error and does not modify "*out".
        //
        // After calling `old_state->Extend()`, `old_state` may no longer be
        // used.
        //
        // NOTE(mrry): This method respects the placement of stateful nodes in
        // in *this, but currently does not transfer any other placement
        // or cost model information to the new graph.
        Status Extend(const GraphDef& extension_def,
                std::unique_ptr<GraphExecutionState>* out) const;
        // Builds a ClientGraph (a sub-graph of the full graph as induced by
        // the Node set specified in "options").  If successful, returns OK
        // and the caller takes the ownership of "*out". Otherwise, returns
        // an error.
        Status BuildGraph(const BuildGraphOptions& options,
                std::unique_ptr<ClientGraph>* out);
        // The graph returned by BuildGraph may contain only the pruned
        // graph, whereas some clients may want access to the full graph.
        const Graph* full_graph() { return graph_; }
        // Returns the node with the given name, or null if it does not exist.
        const Node* get_node_by_name(const string& name) const {
            NodeNameToCostIdMap::const_iterator iter =
                node_name_to_cost_id_map_.find(name);
            if (iter != node_name_to_cost_id_map_.end()) {
                return graph_->FindNodeId(iter->second);
            } else {
                return nullptr;
            }
        }
        // Returns the map of stateful placements as a map of
        // node name to placement string.
        std::unordered_map<string, string> GetStatefulPlacements() const {
            return stateful_placements_;
        }
    private:
        GraphExecutionState(std::unique_ptr<GraphDef>&& graph_def,
                const GraphExecutionStateOptions& options);
        Status InitBaseGraph(std::unique_ptr<Graph>&& graph);
        std::unordered_map<string, string> stateful_placements_;  // Immutable after
        // ctor.
        void SaveStatefulNodes(Graph* graph);
        void RestoreStatefulNodes(Graph* graph);
        // Extract the subset of the graph that needs to be run, adding feed/fetch
        // ops as needed.
        Status PruneGraph(const BuildGraphOptions& options, Graph* graph,
                subgraph::RewriteGraphMetadata* out_rewrite_metadata);

        Status OptimizeGraph(const BuildGraphOptions& options, std::unique_ptr<Graph>* optimized_graph,
                std::unique_ptr<FunctionLibraryDefinition>* optimized_flib);

        // The GraphExecutionState must store a copy of the original GraphDef if
        // either of the following conditions holds:
        //
        // * `session_options_.config.graph_options().place_pruned_graph()` is true.
        // * `session_options_.config.experimental().optimize_for_static_graph()` is
        //   false.
        const std::unique_ptr<GraphDef> original_graph_def_;

        const DeviceSet* device_set_;            // Not owned
        const SessionOptions* session_options_;  // Not owned
        // Unique session identifier. Can be empty.
        string session_handle_;

        // Map from name to Node for the full graph in placed_.
        NodeNameToCostIdMap node_name_to_cost_id_map_;

        // 'flib_def_' is initialized from the initial graph def's library,
        // and may be updated by a graph optimization pass.
        std::unique_ptr<FunctionLibraryDefinition> flib_def_;

        // The dataflow graph owned by this object.
        Graph* graph_;

        // `rewrite_metadata_` is only set for GraphExecutionState
        // objects created by `MakeForPrunedGraph()`.
        // cache
        std::unique_ptr<subgraph::RewriteGraphMetadata> rewrite_metadata_;

        DISALLOW_COPY_AND_ASSIGN(GraphExecutionState);
};

#endif
