#include <iostream>

#include "core/op_converter.h"



// no comma
// DEFINE_OP_CONVERTER(ConvOpConverter)



class ConvOpConverter: public OpConverter{
    public:
        ConvOpConverter(){}
        virtual ~ConvOpConverter(){}
        virtual void Run()override;
};


void ConvOpConverter::Run(){
    std::cout<<"ConvOpConverter"<<std::endl;
}


REGISTER_CLASS_OP(ConvOpConverter);





