#include <memory>
#include <string>
#include <algorithm>
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
    int32_t Node::num_outputs() const { return props_->node_def.input_index().size(); }
    int32_t Node::num_inputs() const { return props_->node_def.input_index().size(); }
    const std::string& Node::name() const { return props_->node_def.name(); }
    const std::string& Node::type_string() const { return props_->node_def.type(); }
    const ::dlxnet::NodeProto& Node::def() const { return props_->node_def; }

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

    void Node::output_edge(int idx, const Edge** e)const{
        if (idx < 0 || idx >= num_outputs()) {
            LOG(FATAL)<<"Invalid output_edge index: "<< idx<< ", Node "<<
                name()<< " only has "<< num_outputs()<<
                " outputs.";
        }

        for (const Edge* edge : out_edges()) {
            if (edge->src_output() == idx) {
                *e = edge;
                return ;
            }
        }
    }

    void Node::input_edge(int idx, const Edge** e) const {
        if (idx < 0 || idx >= num_inputs()) {
            LOG(FATAL)<<"Invalid input_edge index: "<< idx<< ", Node "<<
                name()<< " only has "<< num_inputs()<<
                " inputs.";
        }

        // This does a linear search over the edges.  In the common case,
        // the number of elements is small enough that this search isn't
        // expensive.  Should it become a bottleneck, one can make an
        // optimization where, if the number of edges is small, we use
        // linear iteration, and if the number of edges is large, we perform
        // an indexing step during construction that keeps an array of Edges
        // indexed by pointer.  This would keep the size of each Node small
        // in the common case but make this function faster when the number
        // of edges is large.
        for (const Edge* edge : in_edges()) {
            if (edge->dst_input() == idx) {
                *e = edge;
                return ;
            }
        }

        LOG(FATAL)<<"Could not find input edge "<< idx<< " for "<< name();
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
        // check any edge free exist
        Edge* e = nullptr;
        e = new Edge;

        // populate edge, add it to src node and dst node,
        // then add it to graph
        e->id_ = edges_.size();
        e->src_ = source;
        e->dst_ = dest;
        e->src_output_ = x;
        e->dst_input_ = y;
        CHECK(source->out_edges_.insert(e).second);
        CHECK(dest->in_edges_.insert(e).second);
        edges_.push_back(e);
        ++num_edges_;
        return e;
    }

    std::string Graph::NewName(std::string prefix){
        return "";
    }

    void Graph::RemoveEdge(const Edge* edge){
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

    void Graph::ToGraphDef(::dlxnet::GraphProto* graph_def) const{
        graph_def->Clear();
        graph_def->mutable_node()->Reserve(std::max(1, num_node_ids()));
        std::vector<const Edge*>
            inputs;  // Construct this outside the loop for speed.
        for(int i=0;i<num_node_ids();++i){
            const Node* node = FindNodeId(i);
            auto node_def = graph_def->add_node();
            *node_def = node->def();

            // Get the inputs for this Node.  We make sure control inputs are
            // after data inputs, as required by GraphDef.
            inputs.clear();
            inputs.resize(node->num_inputs(), nullptr);
            for (const Edge* edge : node->in_edges()) {
                inputs[edge->dst_input()] = edge;
            }
            // Sort the control inputs for more predictable serialization.
            std::sort(inputs.begin() + node->num_inputs(), inputs.end(),
                    [](const Edge* a, const Edge* b) -> bool {
                    return a->src()->name() < b->src()->name();
                    });

            node_def->clear_input_index();
            node_def->mutable_input_index()->Reserve(inputs.size());

            for (size_t i = 0; i < inputs.size(); ++i) {
                const Edge* edge = inputs[i];
                const Node* src = edge->src();
                // AddInput(node_def, src->name(), edge->src_output());
            }
        }
    }


    Graph::~Graph(){}
}
