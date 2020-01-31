#ifndef CONVERTER_CORE_OP_CONVERTER_H_
#define CONVERTER_CORE_OP_CONVERTER_H_



class OpConverter{
    OpConverter();
    virtual ~OpConverter();

    // derived class implement this func
    virtual void Run();
};

class OpConverterRegistry{
    static OpConverterRegistry* Global();

    void LookUp();

    void Register();
};

#define OP_CONVETER_REGISTER



#endif
