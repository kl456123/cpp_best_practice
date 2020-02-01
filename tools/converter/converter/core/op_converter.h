#ifndef CONVERTER_CORE_OP_CONVERTER_H_
#define CONVERTER_CORE_OP_CONVERTER_H_
#include "core/registry.h"



class OpConverter: public RegistryItemBase{
    public:
        OpConverter();
        virtual ~OpConverter();

        // derived class implement this func
        virtual void Run()=0;
};


#define REGISTER_CLASS_OP(CLASS)   \
    REGISTER_CLASS(OpConverter, CLASS)

// INSTANIZE_REGISTRY(OpConverter);




#endif
