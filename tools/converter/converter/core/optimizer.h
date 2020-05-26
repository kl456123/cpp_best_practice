#ifndef CORE_OPTIMIZER_H_
#define CORE_OPTIMIZER_H_
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>

#include "graph/graph.h"

class OptimizationPass{
    public:
        virtual void Run(graph::Graph* graph)=0;
        virtual std::string name() const  =0;
};

class Optimizer{
    public:
        static Optimizer* Global();

        void RegisterPass(std::string pass_name,
                OptimizationPass* pass);
        void LookUpPass(const std::string pass_name,
                OptimizationPass** pass)const;
        void Optimize(graph::Graph* graph)const;

    private:
        Optimizer();
        std::unordered_map<std::string, OptimizationPass*> passes_;

};

template <typename Pass>
class RegisterOptimizationPassHelper{
    public:
        RegisterOptimizationPassHelper(
                std::string pass_name){
            Optimizer::Global()->RegisterPass(pass_name, new Pass());
        }
};

#define REGISTER_PASS(Pass)  REGISTER_CLASS_UNIQUE_HELPER(Pass, __COUNTER__)

#define REGISTER_CLASS_UNIQUE_HELPER(pass, ctr) REGISTER_CLASS_UNIQUE(pass, ctr)

#define REGISTER_CLASS_UNIQUE(Pass, ctr)        \
    static auto __reg##Pass##ctr =      \
    RegisterOptimizationPassHelper<Pass>(#Pass)



#endif
