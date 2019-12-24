#include "Pool.h"
#include "test_suite.h"
#include <cstdlib>
#include <ctime>



class PoolTestCase: public TestCase{
    public:
        virtual bool run(){
            Pool<float> pool;

            // do benchmark
            int loop_times = 1000;
            srand(time(NULL));
            int MAX_SIZE=100000;
            for(int i=0;i<loop_times;i++){
                int size = random()%MAX_SIZE;
                float* chunk = pool.alloc(size);
                pool.recycle(chunk);
            }
            pool.clear();
            return false;
        }
};


// register
TestSuiteRegister(PoolTestCase, "PoolTestCase");
