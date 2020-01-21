#include "core/op_builder.h"


namespace{
    void FinalizeAttr(std::string spec, OpDef* op_def, std::vector<std::string>* errors){
        //(TODO: breakpoint, need some strings operator utils to handle strings)
        OpDef::AttrDef* attr = op_def->add_attr();
    }
}

OpDefBuilder::OpDefBuilder(std::string name){
    op_def()->set_name(std::move(name));
}


OpDefBuilder& OpDefBuilder::Attr(std::string spec){
    attrs_.push_back(std::move(spec));
    return *this;
}

OpDefBuilder& OpDefBuilder::Input(std::string spec){
    inputs_.push_back(std::move(spec));
    return *this;
}

OpDefBuilder& OpDefBuilder::Output(std::string spec){
    outputs_.push_back(std::move(spec));
    return *this;
}

OpDefBuilder& OpDefBuilder::SetShapeFn(OpShapeInferenceFn fn){
    if(op_reg_data_.shape_inference_fn!=nullptr){
        errors_.push_back("SetShapeFn called twice for op");
    }else{
        op_reg_data_.shape_inference_fn = OpShapeInferenceFn(fn);
    }
    return *this;
}

Status OpDefBuilder::Finalize(OpRegistrationData* op_reg_data)const{
    // parse all params and store all errors when it happens
    std::vector<std::string> errors = errors_;
    *op_reg_data = op_reg_data_;
    OpDef* op_def = &op_reg_data->op_def;
    for(auto attr: attrs_){
        FinalizeAttr(attr, op_def, &errors);
    }

    for(auto input:inputs_){
    }

    for(auto output: outputs_){

    }
    if(errors.empty()){
        return Status::OK();
    }
    THROW_ERROR("Invalid arguments!\n");
}



