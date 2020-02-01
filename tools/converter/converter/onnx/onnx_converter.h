#ifndef CONVERTER_ONNX_CONVERTER_H_
#define CONVERTER_ONNX_CONVERTER_H_
#include "core/converter.h"



class ONNXConverter: public Converter{
    public:
        ONNXConverter()=default;
        ONNXConverter(const ConverterConfig config);
        virtual ~ONNXConverter(){}
        virtual void Run()override;

        void PrintSelf(){
        }
};


REGISTER_CLASS_CONVERTER(ONNXConverter);

#endif
