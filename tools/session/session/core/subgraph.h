#include <vector>

#include "types.pb.h"
// #include "session/core/subgraph.h"


namespace subgraph{
        struct RewriteGraphMetadata{
            std::vector<DataType> feed_types;
            std::vector<DataType> fetch_types;
        };
};
