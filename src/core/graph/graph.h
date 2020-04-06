#ifndef CORE_GRAPH_GRAPH_H_
#define CORE_GRAPH_GRAPH_H_
#include "core/macros.h"
#include "core/types.h"

class Graph{
    public:
        explicit Graph();
        ~Graph();
        void ToGraphDef(GraphDef* graph_def)const;
    private:
        TF_DISALLOW_COPY_AND_ASSIGN(Graph);
};


#endif
