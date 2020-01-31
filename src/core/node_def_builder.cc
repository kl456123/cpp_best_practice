#include "core/node_def_builder.h"


// struct NodeOut
NodeDefBuilder::NodeOut::NodeOut(std::string n, )
    : node(n), index(i), data_type(dt){}


    NodeDefBuilder::NodeOut::NodeOut(){
        // call reset()
    }

void NodeDefBuilder::NodeOut::Reset(){
    node = n;
    index = i;
    data_type = dt;
}


// NodeDefBuilder

NodeDefBuilder::NodeDefBuilder(std::string name, std::string op_name,
        const OpRegistry* op_registry){
    node_def_.set_name(name);
    const Status status = op_registry->LookUp();
    if(status.ok()){
        Initialize();
    }else{
        errors_.push_back(status.error_description());
        input_specified_=0;
    }
}

NodeDefBuilder::NodeDefBuilder(std::string name, const OpDef* op_def)
    :op_def_(op_def){
        node_def_.set_name(name);
        Initialize();
    }

void NodeDefBuilder::Initialize(){
    input_specified_=0;
    node_def_.set_op(op_def_->name());
}

const OpDef::ArgDef* NodeDefBuilder::NextArgDef(){
    if(!NextArgAvailable())return nullptr;
    return &op_def_->input_arg(inputs_specified_++);
}

bool NodeDefBuilder::NextArgAvailable(){
    if(op_def_==nullptr){
        return false;
    }else if(inputs_specified_>=op_def_->input_arg_size()){
        errors_.push_back(std::string("More Input() calls than the input_args"));
        return false;
    }
    return true;
}

NodeDefBuilder& NodeDefBuilder::Input(const NodeOut& src){
    Input(src.node, src.index, src.data_type);
    return *this;
}

// add "src_node:src_index"
void NodeDefBuilder::AddInput(std::string src_node, int src_index){
    if(src_ndoe.empty()){
        errors_.push_back("Empty input node name");
    }else if(src_node[0]=='^'){
        errors_.push_back(std::);
    }else if(src_index>0){
        node_def_.add_input(src_node+":"+std::string(src_index));
    }else{
        node_def_.add_input(src_node);
    }
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const AttrValue){
}

void NodeDefBuilder::SingleInput(const OpDef::ArgDef* input_arg,
        std::string src_node, int src_index, DataType dt){
    AddInput(src_node, src_index);
    if(input_arg->type()!=DT_INVALID){
        const DataType expected = MaybeAddRef();
        VerifyInput();
    }else{
        VerifyInputRef();
        Attr();
    }
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Input(std::string src_node, int src_index, DataType dt){
    const OpDef::ArgDef* arg = NextArgDef();
    if(arg!=nullptr) SingleInput(arg, );
    return *this;
}
