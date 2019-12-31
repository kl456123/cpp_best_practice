#include "test_suite.h"
#include "core/backend.h"
// #include "backends/cpu/cpu_backend.h"
// #include "backends/opencl/opencl_backend.h"


class CPUBackendTestCase : public TestCase{
    public:
        bool run(){
            auto backend_type = Backend::ForwardType::CPU;
            Backend* backend= ExtractBackend(backend_type);
            int num = 100;
            int buffer_size = num*sizeof(float);
            // float* data_ptr = (float*)(backend->Alloc(buffer_size));
            // for(int i=0;i<100;i++){
                // data_ptr[i] = 1.5;
            // }
            return true;
        }
};


class OpenCLBackendTestCase : public TestCase{
    public:
        bool run(){
            return true;
        }
};


TestSuiteRegister(CPUBackendTestCase, "CPUBackendTestCase");
TestSuiteRegister(OpenCLBackendTestCase, "OpenclBackendTestCase");
