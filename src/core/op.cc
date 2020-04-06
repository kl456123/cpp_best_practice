#include "core/op.h"
#include "stream_executor/platform/errors.h"
#include <memory>


OpRegistry::~OpRegistry(){
    // delete op_reg_data
    for(const auto&e: registry_){
        delete e.second;
    }
}

OpRegistry* OpRegistry::Global(){
    static OpRegistry* global = new OpRegistry();
    return global;
}

void OpRegistry::GetOpRegistrationData(std::vector<OpRegistrationData>* op_reg_datas){
    for(const auto& p:registry_){
        op_reg_datas->push_back(*p.second);
    }
}

void OpRegistry::GetOpRegistrationOps(std::vector<OpDef>* op_defs){
    for(const auto& p:registry_){
        op_defs->push_back(p.second->op_def);
    }
}


Status OpRegistry::Register(const OpRegistrationDataFactory& op_data_factory){
    // status has been updating during total life-time
    std::unique_ptr<OpRegistrationData> op_reg_data(new OpRegistrationData);
    // populate op_reg_data
    Status s = op_data_factory(op_reg_data.get());
    if(s.ok()){
        // insert to map
        registry_.insert(std::make_pair(op_reg_data->op_def.name(), op_reg_data.get()));
    }else{
        s = ::tensorflow::errors::Internal("registered Op with name failed!\n");
    }

    // callback watcher
    Status watcher_status = s;
    if(watcher_){
        watcher_status = watcher_(s, op_reg_data->op_def);
    }

    if(s.ok()){
        op_reg_data.release();
    }else{
        // release memory resource
        op_reg_data.reset();
    }

    return watcher_status;
}

Status OpRegistry::LookUp(const std::string& name, const OpRegistrationData** op_reg_data){
    auto it = registry_.find(name);
    if(it!=registry_.end()){
        *op_reg_data = it->second;
        return Status::OK();
    }

    return ::tensorflow::errors::Internal("cannot find op with name \n");
}

Status OpRegistry::SetWatcher(const Watcher& watcher){
    if(watcher_&&watcher){
        return ::tensorflow::errors::Internal("cannot overwrite a valid watcher!\n");
    }

    watcher_ = watcher;
    return Status::OK();
}



OpDefBuilderReceiver::OpDefBuilderReceiver(const OpDefBuilderWrapper<true>& wrapper){
    // register it in constructor
    //(TODO) closure
    OpRegistry::Global()->Register([wrapper](OpRegistrationData* op_reg_data)->Status{
            return wrapper.builder().Finalize(op_reg_data);
            });
}
