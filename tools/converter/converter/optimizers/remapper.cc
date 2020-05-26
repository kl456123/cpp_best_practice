#include <glog/logging.h>
#include <unordered_set>

#include "optimizers/remapper.h"



namespace optimizer{
    void MergeBatchNormToConvolution(graph::Node* bn_node, graph::Node* conv2d_node){
    }

    void Remapper::Run(graph::Graph* graph){
        // find conv2d bn bias

        std::unordered_set<graph::Node* > nodes_to_delete;
        for(int i=0;i<graph->num_node_ids();++i){
            // find specified pattern from each node
            auto node = graph->FindNodeId(i);

            // ignore when processed already
            if(nodes_to_delete.find(node)!=nodes_to_delete.end()){
                continue;
            }

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
            VLOG(1)<<"exist chance to merge conv and batchnorm here";
            MergeBatchNormToConvolution(next_node, node);
            nodes_to_delete.insert(next_node);
        }

        for(auto node: nodes_to_delete){
            // get src and dst first
            auto src = node->input_edge(0)->src();
            auto dst = node->output_edge(0)->dst();
            auto src_index = node->input_edge(0)->src_output();
            auto dst_index = node->output_edge(0)->dst_input();

            // then remove current node
            graph->RemoveNode(node);

            // finally reconnect prev node and next node
            graph->AddEdge(src, src_index, dst, dst_index);
        }
    }
    REGISTER_PASS_WITH_NAME(Remapper, "Remapper");
}
