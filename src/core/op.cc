#include "core/op.h"

OpRegistry* OpRegistry::Global(){
    static OpRegistry* global = new OpRegistry();
    return global;
}



OpDefBuilderReceiver::OpDefBuilderReceiver(const OpDefBuilderWrapper<true>& wrapper){
    // register it in constructor
    //(TODO) closure
    OpRegistry::Global()->Register([wrapper](OpRegistrationData* op_reg_data)->Status{
            return wrapper.builder().Finalize(op_reg_data);
            });
}
