#include <vector>
#include <string>

#include "core/kernel_builder.h"
#include "core/logging.h"


KernelDefBuilder::KernelDefBuilder(const char* op_name){
    kernel_def_= new KernelDef;
    kernel_def_->set_op(op_name);
}

KernelDefBuilder::~KernelDefBuilder(){
    // it will be nullptr when build() called
    CHECK(kernel_def==nullptr)<<"Did not call Build()\n";
}


KernelDefBuilder& KernelDefBuilder::Device(const char* device_type){
    kernel_def_->set_device_type(device_type);
    return *this;
}



template<>
KernelDefBuilder& KernelDefBuilder::AttrConstraint<int64_t>(
        const char* attr_name, std::vector<int64_t> allowed){
    auto* constraint = kernel_def_->add_constraint();
    constraint->set_name(attr_name);
    auto* allowed_values = constraint->mutable_allowed_values()->mutable_list();
    for(const int64_t integer: allowed){
        LOG(INFO)<< integer;
        allowed_values->add_i(integer);
    }

    return *this;
}

template<>
KernelDefBuilder& KernelDefBuilder::AttrConstraint<int64_t>(const char* attr_name,
        int64_t allowed){
    return AttrConstraint(attr_name, std::vector<int64_t>({allowed}));
}

template<>
KernelDefBuilder& KernelDefBuilder::AttrConstraint<std::string>(const char* attr_name, std::vector<std::string> allowed){
    auto* constraint = kernel_def_->add_constraint();
    constraint->set_name(attr_name);
    auto* allowed_values = constraint->mutable_allowed_values()->mutable_list();
    for(const auto& str:allowed){
        allowed_values->add_s(str);
    }
    return *this;
}

template<>
KernelDefBuilder& KernelDefBuilder::AttrConstraint<std::string>(const char* attr_name, std::string allowed){
    return AttrConstraint<std::string>(attr_name, std::vector<std::string>({allowed}));
}


KernelDefBuilder& KernelDefBuilder::TypeConstraint(const char* attr_name, std::vector<DataType> allowed){
    auto* constraint = kernel_def_->add_constraint();
    constraint->set_name(attr_name);
    auto* allowed_values = constraint->mutable_allowed_values()->mutable_list();
    for(DataType dt: allowed){
        allowed_values->add_type(dt);
    }
    return *this;
}

KernelDefBuilder& KernelDefBuilder::TypeConstraint(const char* attr_name, DataType allowed){
    return TypeConstraint(attr_name, std::vector<DataType>({allowed}));
}

KernelDefBuilder& KernelDefBuilder::Priority(int32_t priority){
    kernel_def_->set_priority(priority);
    return *this;
}


const KernelDef* KernelDefBuilder::Build(){
    KernelDef* r = kernel_def_;
    kernel_def_ = nullptr;
    return r;
}
