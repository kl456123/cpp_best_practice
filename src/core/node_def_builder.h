#ifndef CORE_NODE_DEF_BUILDER_H_
#define CORE_NODE_DEF_BUILDER_H_
#include <string>

#include "types.pb.h"
#include "node_def.pb.h"
#include "core/op.h"

class NodeDefBuilder;


class NodeDefBuilder{
    public:
        struct NodeOut{
            NodeOut(std::string n, int i, DataType dt);
            NodeOut();
            void Reset(std::string n, int i, DataType dt);
            std::string node;
            int index;
            DataType data_type;
        };

        NodeDefBuilder(std::string name, std::string op_name,
                const OpRegistry* op_registry=OpRegistry::Global());
        NodeDefBuilder(std::string name, const OpDef* op_def);

        NodeDefBuilder& Input();

        const OpDef::ArgDef* NextArgDef();

        bool NextArgAvailable();

        NodeDefBuilder& Device(std::string device_spec);

        NodeDefBuilder& Attr();

        Status Finalize(NodeDef* node_def);

        const string& node_name()const{return node_def_.name();}

    private:
        void Initialize();
        const OpDef* op_def_;
        int input_specified_;
        NodeDef* node_def_;
        std::vector<std::string> errors_;
};

#endif
