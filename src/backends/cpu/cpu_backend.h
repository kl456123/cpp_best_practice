#ifndef BACKENDS_CPU_CPU_BACKEND_H_
#define BACKENDS_CPU_CPU_BACKEND_H_
#include <memory>
#include "core/backend.h"
#include "core/pool.h"


class Tensor;

class CPUPool final: public Pool{
    public:
        void* Malloc(size_t size, int alignment)override;
};



class CPUBackend final: public Backend{
    public:
        CPUBackend(Backend::ForwardType type);
        virtual ~CPUBackend();

        void Alloc(Tensor* )override;
        void Clear()override;
        void Recycle(Tensor* )override;

        const CPUPool* pool()const{return mPool.get();}
    private:
        std::shared_ptr<CPUPool> mPool;

};

void RegisterCPUBackend();
#endif
