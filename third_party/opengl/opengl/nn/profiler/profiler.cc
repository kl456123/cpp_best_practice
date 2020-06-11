#include <sstream>
#include "opengl/utils/logging.h"
#include "opengl/core/types.h"

#include "opengl/nn/profiler/profiler.h"
#include "opengl/core/step_stats.pb.h"

namespace opengl{
    void Profiler::CollectData(StepStats* step_stats){
        std::stringstream ss;
        float total_micros = 0;
        for(auto& dev_stats:step_stats->dev_stats()){
            ss<<"device name: "<<dev_stats.device()<<std::endl;
            // loop all nodes which in that device
            for(auto& node_stats: dev_stats.node_stats()){
                ss<<"node name: "<<node_stats.node_name()<<"\t"
                    <<"compute time: "<<node_stats.all_end_rel_micros()*1e-3<<" ms\n";
                total_micros+=node_stats.all_end_rel_micros()*1e-3;
            }
        }
        ss<<"Total Computation Time: "<<total_micros<<" ms\n";
        ss<<"Session Setup Time: "<<step_stats->all_setup_time_micros()*1e-3<<" ms\n";
        ss<<"Output Time: "<<step_stats->output_time_micros()*1e-3<<" ms\n";
        LOG(INFO)<<ss.str();
    }
}
