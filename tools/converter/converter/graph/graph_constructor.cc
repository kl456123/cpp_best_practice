#include "graph_constructor.h"

namespace graph{
    class GraphConstructor{
        public:
            GraphConstructor(::dlxnet::ModelProto&& gdef, Graph* g)
                :graph_def_(gdef),
                g_(g){}
            static bool Construct(::dlxnet::ModelProto&& gdef, Graph* g);
            bool TryImport(){
                // TF_RETURN_IF_ERROR(EnsureNoNameCollisions());
                // TF_RETURN_IF_ERROR(ValidateInputMapAndControlDependencies());
                BuildNodeIndex();
                InitFromEdges();

                // NOTE: Convert() invokes `consume_node_def()` on each node in the input
                // graph, so `get_node_def()` is no longer usable once it is called.
                Convert();

                FixupSourceAndSinkEdges();
                return true;
            }
        private:
            void BuildNodeIndex();
            void InitFromEdges();
            void Convert();
            void FixupSourceAndSinkEdges();

            ::dlxnet::ModelProto graph_def_;
            Graph* g_;
    };

    bool GraphConstructor::Construct(::dlxnet::ModelProto&& gdef, Graph* g){
        GraphConstructor c(std::move(gdef), g);
        return c.TryImport();
    }

    void GraphConstructor::BuildNodeIndex(){
    }

    void GraphConstructor::InitFromEdges(){
    }

    void GraphConstructor::Convert(){
    }

    void GraphConstructor::FixupSourceAndSinkEdges(){
    }

    bool ConvertGraphDefToGraph(const ::dlxnet::ModelProto& gdef, Graph* g) {
        return ConvertGraphDefToGraph(::dlxnet::ModelProto(gdef), g);
    }


    bool ConvertGraphDefToGraph(::dlxnet::ModelProto&& gdef, Graph* g){
        return GraphConstructor::Construct(std::move(gdef), g);
    }
}

