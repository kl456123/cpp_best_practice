#ifndef CORE_OPTIMIZER_H_
#define CORE_OPTIMIZER_H_
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>

#include "graph/graph.h"

class OptimizationPass{
    public:
        virtual void Run(std::unique_ptr<graph::Graph>* graph)=0;
};

class Optimizer{
    public:
        static Optimizer* Global();

        void RegisterPass(std::string pass_name,
        OptimizationPass* pass);
        void LookUpPass(const std::string pass_name,
                OptimizationPass** pass)const;
        void Optimize(std::unique_ptr<graph::Graph>* graph)const;

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

#define REGISTER_PASS(pass_name, Pass)  \
    static auto __reg##pass_name =      \
    RegisterOptimizationPassHelper<Pass>(pass_name)


#endif
