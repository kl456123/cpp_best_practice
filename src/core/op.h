#ifndef CORE_OP_H_
#define CORE_OP_H_



// template class to selet between true and false
template<bool should_register>
class OpDefBuilderWrapper;


// specialize template class when true
template<>
class OpDefBuilderWrapper<true>{
    public:
        explicit OpDefBuilderWrapper(){}

};
// specialize template class when false(do nothing)
template<>
class OpDefBuilderWrapper<false>{
    public:
        explicit OpDefBuilderWrapper(){}

};

// seletive register op
#define REGISTER_OP(name) name






#endif
