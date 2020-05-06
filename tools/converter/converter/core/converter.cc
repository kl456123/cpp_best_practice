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
