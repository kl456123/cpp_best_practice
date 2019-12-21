#include "backend.h"
#include <mutex>


using namespace std;

void register_cpu();

void register_backends(){
    static once_flag of;
    call_once(of, [&]{register_cpu();});
    // register_cpu();

}
static map<string, shared_ptr<Backend>>& get_backends_map(){
    static map<string, shared_ptr<Backend>> backends_map;
    return backends_map;
}

Backend* extract_backend(string name){
    register_backends();
    auto& backends_map = get_backends_map();
    auto iter = backends_map.find(name);
    if(iter==backends_map.end()){
        return nullptr;
    }
    return iter->second.get();
}

void insert_backend(string& name, shared_ptr<Backend>& backend){
    auto& backends_map = get_backends_map();
    backends_map.insert(make_pair(name, backend));
}
