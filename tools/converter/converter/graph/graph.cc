#include <memory>
#include <string>
#include <glog/logging.h>

#include "graph/graph.h"


namespace graph{
    struct NodeProperties{
        NodeProperties(::dlxnet::NodeProto node_def, const std::string op_type)
            :node_def(std::move(node_def)), op_type(op_type){}

        ::dlxnet::NodeProto node_def;
        const std::string op_type;
    };

    Node::Node()
        : id_(-1),
        class_(NC_UNINITIALIZED),
        props_(nullptr){}

    void Node::Initialize(int id,  std::shared_ptr<NodeProperties> props){
        DCHECK_EQ(id_, -1);
        DCHECK(in_edges_.empty());
        DCHECK(out_edges_.empty());
        id_ = id;

        props_ = std::move(props);
        // Initialize the class_ based on the type string
        // TODO(breakpoint) specify which node class to be used for op
        // class_ = GetNodeClassForOp(props_->node_def.op());
    }

    Graph::Graph(){
        // specify opset for the graph
        // where "opset" means system defined operators
    }

    Node* Graph::AddNode(::dlxnet::NodeProto node_def){
        // TODO(breakpoint) use kernel or kernel_type?
        const std::string op_type = node_def.type();
        Node* node = AllocateNode(std::make_shared<NodeProperties>(node_def, op_type));
        return node;
    }

    void Graph::RemoveNode(Node* node){
    }

    const Edge* Graph::AddEdge(Node* source, int x, Node* dest, int y){
        return nullptr;
    }

    std::string Graph::NewName(std::string prefix){
        return "";
    }

    Node* Graph::AllocateNode(std::shared_ptr<NodeProperties> props){
        Node* node;
        node = new Node;

        // initialize node here
        const int id = nodes_.size();
        node->graph_ = this;
        node->Initialize(id, std::move(props));
        nodes_.push_back(node);
        ++num_nodes_;
        return node;
    }


    Graph::~Graph(){}
}
