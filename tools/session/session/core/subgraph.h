#include "session/core/subgraph.h"


namespace subgraph{
        struct RewriteGraphMetadata{
            std::vector<DataType> feed_types;
            std::vector<DataType> fetch_types;
        };
};
