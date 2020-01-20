#ifndef CORE_OP_H_
#define CORE_OP_H_
#include "core/op.h"
#include "core/op_builder.h"
#include "core/seletive_registration.h"

class OpRegistry{
    public:
        static OpRegistry* Global();
        //lazy register(only constructor it when used)
        Register();
};



// template class to selet between true and false
template<bool should_register>
class OpDefBuilderWrapper;


// specialize template class when true
template<>
class OpDefBuilderWrapper<true>{
    public:
        explicit OpDefBuilderWrapper(const char name[]):builder_(name){}
        OpDefBuilderWrapper<true>& Attr(std::string spec){
            builder_.Attr(std::move(spec));
            return *this;
        }

        OpDefBuilderWrapper<true>& Input(std::string spec){
            builder_.Input(std::move(spec));
            return *this;
        }

        OpDefBuilderWrapper<true>& Output(std::string spec){
            builder_.Output(std::move(spec));
            return *this;
        }

        OpDefBuilderWrapper<true>& SetShapeFn(OpShapeInferenceFn fn){
            builder_.SetShapeFn(fn);
            return *this;
        }

        const OpDefBuilder& builder()const{return builder_;}
    private:
        OpDefBuilder builder_;

};
// specialize template class when false(do nothing)
template<>
class OpDefBuilderWrapper<false>{
    public:
        explicit OpDefBuilderWrapper(const char name[]){}
        OpDefBuilderWrapper<false>& Attr(std::string spec){
            return *this;
        }

        OpDefBuilderWrapper<false>& Input(std::string spec){
            return *this;
        }

        OpDefBuilderWrapper<false>& Output(std::string spec){
            return *this;
        }

        OpDefBuilderWrapper<false>& SetShapeFn(OpShapeInferenceFn fn){
            return *this;
        }

};


class OpDefBuilderReceiver{
    public:
        // true
        OpDefBuilderReceiver(const OpDefBuilderWrapper<true>& wrapper);
        //false(do nothing)
        OpDefBuilderReceiver(const OpDefBuilderWrapper<false>& wrapper){
        }
};
// seletive register op
#define REGISTER_OP(name) REGISTER_OP_UNIQ(__COUNTER__, name)

#define REGISTER_OP_UNIQ(ctr, name)                         \
    OpDefBuilderReceiver register_op##ctr =             \
    OpDefBuilderWrapper<SHOULD_REGISTER_OP(name)>(name)






#endif
