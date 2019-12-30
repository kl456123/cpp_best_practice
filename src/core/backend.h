#ifndef CORE_BACKEND_H_
#define CORE_BACKEND_H_

#include <iostream>
#include <map>
#include <string>
#include <memory>

using namespace std;

class Tensor;

class Backend{

    public:
        enum ForwardType{
            CPU,
            OPENCL,
        };
        Backend(ForwardType type):mType(type){
        }
        virtual ~Backend(){}


        virtual void Alloc(Tensor* tensor)=0;
        virtual void Recycle(Tensor* tensor)=0;
        virtual void Clear()=0;

        virtual void CopyFromHostToDevice(Tensor* tensor){};

    private:
        ForwardType mType;

};
void InsertBackend(Backend::ForwardType type_name, shared_ptr<Backend>& backend);
Backend* ExtractBackend(Backend::ForwardType type_name);

#endif
