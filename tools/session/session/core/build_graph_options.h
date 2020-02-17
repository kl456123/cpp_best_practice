#ifndef SESSION_CORE_BUILD_GRAPH_OPTIONS_H_
#define SESSION_CORE_BUILD_GRAPH_OPTIONS_H_
#include <cstdint>
#include <cstddef>
#include <string>

#include "session/core/collective_order.h"

#include "config.pb.h"

typedef int64_t int64;
using std::string;

struct BuildGraphOptions {
    CallableOptions callable_options;

    // If `true`, uses Arg/Retval to implement feeds/fetches; otherwise
    // uses Recv/Send to implement feeds/fetches.
    // TODO(mrry): Remove this when the distributed runtime supports Arg/Retval.
    bool use_function_convention = false;

    static const int64 kNoCollectiveGraphKey = 0;
    int64 collective_graph_key = kNoCollectiveGraphKey;

    // If not `kNone`, order all CollectiveReduce operations statically and
    // deterministically.  If `kEdges`, encode dependencies as explicit control
    // edges, if `kAttrs` encode as attribute on collective op.
    GraphCollectiveOrder collective_order = GraphCollectiveOrder::kNone;

    string DebugString() const;
};

#endif
