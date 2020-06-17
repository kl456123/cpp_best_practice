#ifndef OPENGL_NN_PROFILER_PROFILER_H_
#define OPENGL_NN_PROFILER_PROFILER_H_
#include <vector>
#include <map>

#include "opengl/core/types.h"

namespace opengl{
    class StepStats;
    class Profiler{
        public:
            struct StepStatsTime{
                float total_micros=0;
                float setup_micros=0;
                float output_micros=0;

                std::map<string, std::pair<float, int>> type2time;
            };
            void CollectData(StepStats* step_stats);

            void PrintProfiling();
        private:
            std::vector<StepStatsTime> steps_stats_time_;
    };
}


#endif
