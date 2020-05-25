#include <glog/logging.h>

#include "optimizers/remapper.h"



namespace optimizer{
    void Remapper::Run(graph::Graph* graph){
        // find conv2d bn bias
        for(int i=0;i<graph->num_node_ids();++i){
            // find specified pattern from each node
            auto node = graph->FindNodeId(i);
            if(node->type_string()!="Conv"){
                continue;
            }
            // fall to conv case
            const graph::Edge* e=nullptr;
            node->output_edge(0, &e);
            graph::Node* next_node = e->dst();
            // check the next node
            if(next_node->type_string()!="BatchNormalization"){
                continue;
            }
            // conv + bn
            // merge them
            LOG(INFO)<<"exist chance to merge conv and batchnorm here";
        }
    }
}
