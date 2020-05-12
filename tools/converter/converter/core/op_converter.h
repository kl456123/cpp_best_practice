#ifndef CONVERTER_CORE_OP_CONVERTER_H_
#define CONVERTER_CORE_OP_CONVERTER_H_
#include <glog/logging.h>

#include "core/registry.h"
#include "dlcl.pb.h"



class OpConverter{
    public:
        OpConverter();
        virtual ~OpConverter();

        // derived class implement this func
        virtual void Run(dlxnet::NodeProto* dst_node, const void* src_node)=0;

        // TODO(breakpoint) add description here
        virtual void AddInputConstant(dlxnet::GraphProto* graph,
                std::unordered_map<std::string, int>* total_tensor_names);
};


#define REGISTER_CLASS_OP(CLASS)   \
    REGISTER_CLASS(OpConverter, CLASS)

#define REGISTER_OP_WITH_NAME(CLASS, name)  \
    REGISTER_CLASS_BY_NAME(OpConverter, name, CLASS)


// INSTANIZE_REGISTRY(OpConverter);




#endif
