#ifndef CONVERTER_ONNX_CONVERTER_H_
#define CONVERTER_ONNX_CONVERTER_H_
#include "core/converter.h"
#include "onnx.pb.h"



class ONNXConverter: public Converter{
    public:
        ONNXConverter()=default;
        ONNXConverter(const ConverterConfig config);
        virtual ~ONNXConverter(){}
        virtual void Run()override;

        void PrintSelf(){}
        void MakeTensorFromProto(const onnx::TensorProto&, TensorProto*);
};


REGISTER_CLASS_CONVERTER(ONNXConverter);

#endif
