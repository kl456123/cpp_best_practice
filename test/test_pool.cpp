#include "core/pool.h"
#include "test/test_suite.h"
#include <cstdlib>
#include <memory>
#include <ctime>



class PoolTestCase: public TestCase{
    public:
        virtual bool run(){
            std::shared_ptr<Pool> pool;

            // do benchmark
            int loop_times = 1000;
            srand(time(NULL));
            int MAX_SIZE=100000;
            for(int i=0;i<loop_times;i++){
                int size = random()%MAX_SIZE;
                float* chunk = pool->Alloc<float>(size);
                pool->Recycle<float>(chunk);
            }
            pool->Clear();
            return false;
        }
};


// register
TestSuiteRegister(PoolTestCase, "PoolTestCase");
