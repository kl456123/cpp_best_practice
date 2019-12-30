#include "backends/cpu/cpu_backend.h"
#include "test_suite.h"
#include <cstdlib>
#include <memory>
#include <ctime>



class PoolTestCase: public TestCase{
    public:
        virtual bool run(){
            std::shared_ptr<Pool> pool(new CPUPool);

            // do benchmark
            int loop_times = 1000;
            srand(time(NULL));
            int MAX_SIZE=100000;
            for(int i=0;i<loop_times;i++){
                int size = random()%MAX_SIZE;
                void* chunk = pool->Alloc(size*sizeof(float));
                pool->Recycle(chunk);
            }
            pool->Clear();
            return false;
        }
};


// register
TestSuiteRegister(PoolTestCase, "PoolTestCase");
