#include <unordered_map>
#include "core/op_kernel.h"


struct KernelRegistration{
    KernelRegistration(const KernelDef& d, std::string c, std::unique_ptr<kernel_factory::OpKernelFactory>f)
        :def(d), kernel_class_name(c), factory(std::move(f)){}
    const KernelDef def;
    const std::string kernel_class_name;
    std::unique_ptr<kernel_factory::OpKernelFactory> factory;
};

struct KernelRegistry{
    // multimap
    std::unordered_multimap<std::string, KernelRegistration> registry;
};

void* GlobalKernelRegistry(){
    static KernelRegistry* global_kernel_registry = new KernelRegistry;
    return global_kernel_registry;
}

namespace kernel_factory{
    void OpKernelRegistrar::InitInternal(const KernelDef* kernel_def,
            std::string kernel_class_name, std::unique_ptr<OpKernelFactory> factory){
        if(kernel_def->op()!="_no_register"){
            // get key
            const string_key = Key(kernel_def->op(), DeviceType(kernel_def->device_type()));

            auto global_registry = reinterpret_cast<KernelRegistry*>(GlobalKernelRegistry());
            global_registry->registry.emplace(
                    key,
                    KernelRegistration(*kernel_def, kernel_class_name, std::move(factory)));
        }
        delete kernel_def;
    }

    // implement Create func
    OpKernel* OpKernelRegistrar::PtrOpKernelFactory::Create(){
        // call create_func_ here using context argument
        return (*create_func_)(context);
    }
}//namespace kernel_factory

const string& OpKernel::requested_device(){
}

// OpKernelContext
void OpKernelContext::CtxFailure(const char* file, int line, const Status& s){
    LOG(1)<<"OP_REQUIRES failed at "<<file<< ":"<<line<<" : "<<s;
    SetStatus(s);
}

void OpKernelContext::CtxFailureWithWarning(const char* file, int line, const Status& s){
    LOG(WARNING)<<"OP_REQUIRES failed at "<<file<<":"<<line<< " : "<<s;
    SetStatus(s);
}

void OpKernelContext::CtxFailure(const Status& s){
    LOG(1)<<s;
    SetStatus(s);
}

void OpKernelContext::CtxFailureWithWarning(const Status& s){
    LOG(WARNING)<<s;
    SetStatus(s);
}

// OpKernelConstruction
void OpKernelConstruction::CtxFailure(const Status& s) {
  LOG(1) << s;
  SetStatus(s);
}

void OpKernelConstruction::CtxFailureWithWarning(const Status& s) {
  LOG(WARNING) << s;
  SetStatus(s);
}

void OpKernelConstruction::CtxFailure(const char* file, int line,
                                      const Status& s) {
  LOG(1) << "OP_REQUIRES failed at " << file << ":" << line
          << " : " << s;
  SetStatus(s);
}

void OpKernelConstruction::CtxFailureWithWarning(const char* file, int line,
                                                 const Status& s) {
  LOG(WARNING) << "OP_REQUIRES failed at " << file << ":" << line
               << " : " << s;
  SetStatus(s);
}
