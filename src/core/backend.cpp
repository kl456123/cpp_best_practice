#include "backend.h"
#include <mutex>


using namespace std;

void RegisterCPUBackend();
void RegisterOpenCLBackend();

void RegisterBackend(){
    static once_flag of;
    call_once(of, [&]{
            RegisterCPUBackend();
            RegisterOpenCLBackend();
            });
}
static map<Backend::ForwardType, shared_ptr<Backend>>& backends_map(){
    static map<Backend::ForwardType, shared_ptr<Backend>> s_backends_map;
    return s_backends_map;
}

Backend* ExtractBackend(Backend::ForwardType type_name){
    RegisterBackend();
    auto& s_backends_map = backends_map();
    auto iter = s_backends_map.find(type_name);
    if(iter==s_backends_map.end()){
        return nullptr;
    }
    return iter->second.get();
}

void InsertBackend(Backend::ForwardType type, shared_ptr<Backend>& backend){
    auto& s_backends_map = backends_map();
    s_backends_map.insert(make_pair(type, backend));
}
