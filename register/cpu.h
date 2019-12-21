#ifndef REGISTER_CPU_H_
#define REGISTER_CPU_H_
#include "backend.h"
#include <iostream>
#include <string>
#include <memory>
using namespace std;

class CPU: public Backend{
    public:
        CPU():Backend(){}
};

#endif
