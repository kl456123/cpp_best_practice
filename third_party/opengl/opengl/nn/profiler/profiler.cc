#include <sstream>
#include <map>
#include "opengl/utils/logging.h"
#include "opengl/core/types.h"

#include "opengl/nn/profiler/profiler.h"
#include "opengl/core/step_stats.pb.h"

namespace opengl{
    void Profiler::CollectData(StepStats* step_stats){
        std::stringstream ss;
        float total_micros = 0;
        // process each node time according to its kernel type
        // name: nsec, count
        std::map<string, std::pair<float, int>> type2time;
        for(auto& dev_stats:step_stats->dev_stats()){
            ss<<"device name: "<<dev_stats.device()<<std::endl;
            // loop all nodes which in that device
            for(auto& node_stats: dev_stats.node_stats()){
                const float t = node_stats.all_end_rel_micros()*1e-3;
                // ss<<"node name: "<<node_stats.node_name()<<"\t"
                // <<"compute time: "<<t<<" ms\n";
                total_micros+=t;
                if(type2time.find(node_stats.node_type())!=type2time.end()){
                    type2time[node_stats.node_type()].first+=t;
                    type2time[node_stats.node_type()].second++;
                }else{
                    type2time[node_stats.node_type()].first = t;
                    type2time[node_stats.node_type()].second = 1;
                }
            }
        }
        ss<<"Total Computation Time: "<<total_micros<<" ms\n";
        ss<<"Session Setup Time: "<<step_stats->all_setup_time_micros()*1e-3<<" ms\n";
        ss<<"Output Time: "<<step_stats->output_time_micros()*1e-3<<" ms\n";

        // print time map
        ss<<"Node Type\tTime\tCount\n";
        for(auto& iter:type2time){
            ss<<iter.first<<"\t\t"
                <<iter.second.first<<"ms\t\t"
                <<iter.second.second<<"\n";
        }
        std::cout<<ss.str()<<std::endl;
    }
}
