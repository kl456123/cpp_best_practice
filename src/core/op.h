#ifndef CORE_OP_H_
#define CORE_OP_H_
#include <functional>
#include <unordered_map>
#include "core/op.h"
#include "core/op_builder.h"
#include "core/selective_registration.h"

class OpRegistry{
    public:
        OpRegistry()=default;
        virtual ~OpRegistry();
        static OpRegistry* Global();
        // function used to decorate OpRegistrationData
        typedef std::function<Status(OpRegistrationData* )> OpRegistrationDataFactory;
        //only registger op_reg_data, but pass factory func to register func
        //populate op_reg_data in the inner of Register
        Status Register(const OpRegistrationDataFactory& op_data_factory);
        // find item by name
        // change pointer value, so use void**
        Status LookUp(const std::string& name, const OpRegistrationData** op_reg_data);

        // get total
        void GetOpRegistrationData(std::vector<OpRegistrationData>* op_reg_datas);
        void GetOpRegistrationOps(std::vector<OpDef>* op_defs);
        typedef std::function<Status(const Status&, const OpDef&)> Watcher;

        // register watcher
        Status SetWatcher(const Watcher& watcher);

    private:
        mutable std::unordered_map<std::string, const OpRegistrationData*> registry_;
        mutable Watcher watcher_;
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
    OpDefBuilderReceiver register_op##ctr =                 \
    OpDefBuilderWrapper<SHOULD_REGISTER_OP(name)>(name)

// force register op
#define REGISTER_OP_FORCE(name) REGISTER_OP_FORCE_UNIQ(__COUNTER__, name)

#define REGISTER_OP_FORCE_UNIQ(ctr, name)               \
    OpDefBuilderReceiver register_op##ctr =             \
    OpDefBuilderWrapper<true>(name)






#endif
