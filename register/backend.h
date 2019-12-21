#ifndef REGISTER_BACKEND_H_
#define REGISTER_BACKEND_H_

#include <iostream>
#include <map>
#include <string>
#include <memory>

using namespace std;

class Backend{
    public:
        Backend(){}
};
void insert_backend(string& name, shared_ptr<Backend>& backend);
Backend* extract_backend(string name);
#endif
