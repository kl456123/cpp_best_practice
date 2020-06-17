#include "opengl/nn/profiler/host_tracer_utils.h"
#include "opengl/nn/profiler/profiler_interface.h"
#include "opengl/nn/profiler/traceme.h"
#include "opengl/utils/env.h"
#include "opengl/test/test.h"

namespace opengl{
    namespace profiler{
        std::unique_ptr<ProfilerInterface> CreateHostTracer(
                const ProfilerOptions& options);

        namespace {
            TEST(HostTracerTest, CollectsTraceMeEvents) {
                int32 thread_id = Env::Default()->GetCurrentThreadId();

                auto tracer = CreateHostTracer(ProfilerOptions());

                tracer->Start();
                { TraceMe traceme("hello"); }
                { TraceMe traceme("world"); }
                { TraceMe traceme("contains#inside"); }
                { TraceMe traceme("good#key1=value1#"); }
                { TraceMe traceme("morning#key1=value1,key2=value2#"); }
                { TraceMe traceme("incomplete#key1=value1,key2#"); }
                tracer->Stop();
            }
        }//namespace
    }//namespace profier
}//namespace opengl


