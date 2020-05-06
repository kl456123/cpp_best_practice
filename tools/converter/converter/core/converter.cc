#include "core/converter.h"



Converter::Converter(const ConverterConfig config)
    :converter_config_(config),model_(new Model()){
        model_->set_producer_name("ONNX");
        model_->set_version("0.1");
        model_->set_doc_string("ignored");
    }

void Converter::Save(std::string checkpoint_path){
    // save to ckpt path
}
