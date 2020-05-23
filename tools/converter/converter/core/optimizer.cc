#include "core/optimizer.h"

Optimizer::Optimizer(){
}

void Optimizer::RegisterPass(){
}
void Optimizer::LookUpPass(){
}

/*static*/ Optimizer* Optimizer::Global(){
    return new Optimizer();
}



