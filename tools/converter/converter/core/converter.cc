#include<fstream>
#include <glog/logging.h>

#include "core/converter.h"



void Converter::Save(std::string checkpoint_path){
    CHECK_NOTNULL(model_);
    // save to ckpt path
    std::fstream output(checkpoint_path, std::ios::out
            | std::ios::trunc | std::ios::binary);
    model_->SerializeToOstream(&output);
    LOG(INFO)<<"Save to "<<checkpoint_path<<" Done!";
}


std::string Converter::DebugString()const{
    std::string ret_str;
    ret_str+="ConverterConfig: ";
    ret_str+=converter_config_.src_model_path;
    ret_str+="->";
    ret_str+=converter_config_.dst_model_path;
    ret_str+="\n";

    ret_str+="ModelInfo: ";
    ret_str+=model_->DebugString();
    return ret_str;
}

void Converter::Optimize(const Optimizer* optimizer){
}
