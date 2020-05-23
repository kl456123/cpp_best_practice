#ifndef CORE_OPTIMIZER_H_
#define CORE_OPTIMIZER_H_
#include <vector>
#include <string>

class OptimizationPass{
};

class Optimizer{
    public:
        static Optimizer* Global();

        void RegisterPass();
        void LookUpPass();

    private:
        Optimizer();
        std::vector<OptimizationPass*> passes_;

};

template <typename Pass>
class RegisterOptimizationPassHelper{
    public:
        RegisterOptimizationPassHelper(
                std::string pass_name){
            Optimizer::Global()->RegisterPass(new Pass());
        }
};

#define REGISTER_PASS(pass_name, Pass)  \
    static auto __reg##pass_name =      \
    RegisterOptimizationPassHelper<Pass>(pass_name)


#endif
