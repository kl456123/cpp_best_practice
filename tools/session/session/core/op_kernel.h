#ifndef SESSION_CORE_OP_KERNEL_H_
#define SESSION_CORE_OP_KERNEL_H_

class OpKernelContext;

class OpKernel{
    public:
        virtual void Compute(OpKernelContext* context);
};


#endif
