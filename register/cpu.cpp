#include "cpu.h"


void register_cpu(){
    shared_ptr<Backend> ptr;
    ptr.reset(new CPU());
    auto name = string("CPU");
    insert_backend(name, ptr);
}



