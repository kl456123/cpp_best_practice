#ifndef GRAPH_GRAPH_CONSTRUCTOR_H_
#define GRAPH_GRAPH_CONSTRUCTOR_H_
#include "graph.h"

namespace graph{
    // construct from empty graph(sink and source node)
      bool ConvertGraphDefToGraph(const ::dlxnet::ModelProto& gdef, Graph* g);
      bool ConvertGraphDefToGraph(::dlxnet::ModelProto&& gdef, Graph* g);
}// namespace graph


#endif
