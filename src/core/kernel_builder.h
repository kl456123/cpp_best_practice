#ifndef CORE_KERNEL_BUILDER_H_
#define CORE_KERNEL_BUILDER_H_
#include "core/define.h"

// forward declare(proto type)
class KernelDef;

class KernelDefBuilder{
    public:
        KernelDefBuilder(const char* op_name);
        virtual ~KernelDefBuilder();

        KernelDefBuilder& Device(const char* device_type);

        // list attrs
        template<typename T>
            KernelDefBuilder& AttrConstraint(const char* attr_name, std::vector<T> allowed);
        // single attr
        template<typename T>
            KernelDefBuilder& AttrConstraint(const char* attr_name, T allowed);



        // list type
        KernelDefBuilder& TypeConstraint(const char* attr_name, std::vector<DataType> allowed);

        // single type
        KernelDefBuilder& TypeConstraint(const char* attr_name, DataType allowed);

        KernelDefBuilder& Priority(int32_t priority);

        const KernelDef* Build();

    private:
        KernelDef* kernel_def_;
        DISALLOW_COPY_AND_ASSIGN(KernelDefBuilder);
};




#endif
