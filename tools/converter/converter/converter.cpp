#include <iostream>

#include "core/config.h"
#include "core/converter.h"


int main(int argc,char** argv){
    // get converter_config from parser
    // here we just assign it manualy
    ConverterConfig converter_config;
    converter_config.src_model_path = "../assets/demo.onnx";
    converter_config.dst_model_path = "demo.tmp";
    converter_config.src = ConverterConfig::MODEL_SOURCE::ONNX;

    auto converter_registry = Registry<Converter>::Global();
    std::string format_name = "ONNXConverter";
    Converter* converter=nullptr;
    converter_registry->LookUp(format_name, &converter);

    // init converter
    if(converter==nullptr){
        std::cout<<"Cannot find "<<std::endl;
    }else{
        converter->Reset(converter_config);
        converter->Run();
    }

    return 0;
}