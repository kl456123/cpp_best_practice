#ifndef OPENGL_NN_PROFILER_PROFILER_H_
#define OPENGL_NN_PROFILER_PROFILER_H_

namespace opengl{
    class StepStats;
    class Profiler{
        public:
            void CollectData(StepStats* step_stats);
    };
}


#endif
