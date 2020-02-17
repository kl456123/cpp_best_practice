#ifndef SESSION_CORE_GRAPH_PARTITION_H_
#define SESSION_CORE_GRAPH_PARTITION_H_
#include <functional>
#include <string>

#include "session/core/graph.h"

struct PartitionOptions{
    typedef std::function<std::string(const Node*)> NodeToLocFunc;
    NodeToLocFunc node_to_loc = nullptr;

    // A function that returns a unique graph node name with the given
    // prefix.
    typedef std::function<string(const string&)> NewNameFunc;
    NewNameFunc new_name = nullptr;
};

// Partition "input" graph into a set of graphs, one per location.
// The location for node n is derived by calling opts.node_to_loc(n).
// New nodes added by Partition use "opts.new_name(old_name)" to
// generate node names.
//
// Stores the partitions in *partitions.
Status Partition(const PartitionOptions& opts, Graph* input,
        std::unordered_map<string, GraphDef>* partitions);

// Add control edges to the partitions to control the ordering
// and timing of the recv nodes based on the start times calculated
// using some scheduling algorithm.
Status AddControlEdges(const PartitionOptions& opts,
        std::unordered_map<string, GraphDef>* partitions);


#endif
