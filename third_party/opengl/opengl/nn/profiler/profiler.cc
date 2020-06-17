#include <sstream>
#include <map>
#include "opengl/utils/logging.h"

#include "opengl/nn/profiler/profiler.h"
#include "opengl/core/step_stats.pb.h"

namespace opengl{
    void Profiler::CollectData(StepStats* step_stats){
        float total_micros = 0;
        // process each node time according to its kernel type
        // name: nsec, count
        StepStatsTime step_stats_time;
        auto& type2time = step_stats_time.type2time;
        for(auto& dev_stats:step_stats->dev_stats()){
            VLOG(1)<<"device name: "<<dev_stats.device();
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

        step_stats_time.total_micros = total_micros;
        step_stats_time.setup_micros = step_stats->all_setup_time_micros()*1e-3;
        step_stats_time.output_micros = step_stats->output_time_micros()*1e-3;
        steps_stats_time_.emplace_back(std::move(step_stats_time));
    }

    void Profiler::PrintProfiling(){
        std::stringstream ss;
        StepStatsTime mean_step_stats_time;
        // calc mean of total stats
        // sum first
        const int step_size = steps_stats_time_.size();
        for(auto & step_stats_time: steps_stats_time_){
            mean_step_stats_time.total_micros+=step_stats_time.total_micros;
            mean_step_stats_time.setup_micros+=step_stats_time.setup_micros;
            mean_step_stats_time.output_micros+=step_stats_time.output_micros;
            // mean for time map
            auto& mean_type2time = mean_step_stats_time.type2time;
            auto& type2time = step_stats_time.type2time;
            if(mean_type2time.empty()){
                mean_type2time = type2time;
                continue;
            }
            for(auto iter: type2time){
                mean_type2time[iter.first].first+=iter.second.first;
                mean_type2time[iter.first].second+=iter.second.second;
            }
        }

        ss<<"Total Computation Time: "<<mean_step_stats_time.total_micros/step_size<<" ms\n";
        ss<<"Session Setup Time: "<<mean_step_stats_time.setup_micros/step_size <<" ms\n";
        ss<<"Output Time: "<<mean_step_stats_time.output_micros/step_size<<" ms\n";

        // print time map
        ss<<"Node Type\tTime\tCount\n";
        for(auto& iter:mean_step_stats_time.type2time){
            ss<<iter.first<<"\t\t"
                <<iter.second.first/step_size<<"ms\t\t"
                <<iter.second.second/step_size<<"\n";
        }

        std::cout<<ss.str()<<std::endl;
    }
}
