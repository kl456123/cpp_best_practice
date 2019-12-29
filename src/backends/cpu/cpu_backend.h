#ifndef BACKENDS_CPU_CPU_BACKEND_H_
#define BACKENDS_CPU_CPU_BACKEND_H_
#include <memory>
#include "core/backend.h"


class Pool;
class Tensor;



class CPUBackend: public Backend{
    public:
        CPUBackend();
        virtual ~CPUBackend();

        void Alloc(const Tensor* );
        void Clear();
        void Recycle(const Tensor* );

        const Pool* pool()const{return mPool.get();}
    private:
        std::shared_ptr<Pool> mPool;

};

void RegisterCPUBackend();
#endif
