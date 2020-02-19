#include "session/core/graph.h"
#include "op_def.pb.h"

struct NodeProperties {
 public:
  NodeProperties(const OpDef* op_def, NodeDef node_def,
                 const std::vector<DataType> inputs, const std::vector<DataType> outputs)
      : op_def(op_def),
        node_def(std::move(node_def)),
        input_types(inputs.begin(), inputs.end()),
        output_types(outputs.begin(), outputs.end()) {}

  const OpDef* op_def;  // not owned
  NodeDef node_def;
  const std::vector<DataType> input_types;
  const std::vector<DataType> output_types;
};

Graph::~Graph(){
}

Graph::Graph(OpRegistryInterface* ){
}
