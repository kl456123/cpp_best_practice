#include <memory>
#include <vector>
#include <string>

#include "session/utils/status.h"
#include "session/core/session.h"
#include "session/core/session_options.h"
#include "session/utils/logging.h"

Status LoadGraph(const std::string& graph_file_name,
        std::unique_ptr<Session>* session){
    Session* session_ptr=nullptr;
    SessionOptions options;
    Status create_session_status = NewSession(options, &session_ptr);
    if(!create_session_status.ok()){
        return create_session_status;
    }
    session->reset(session_ptr);

    GraphDef graph_def;

    Status create_graph_status = (*session)->Create(graph_def);
    if(!create_graph_status.ok()){
        return create_graph_status;
    }
    return Status::OK();
}

void InitMain(){
}

int main(){
    // init global environment variables
    InitMain();

    // create session
    std::unique_ptr<Session> session;
    std::string graph_file_name("demo.dlc");
    auto load_graph_status = LoadGraph(graph_file_name, &session);
    if(!load_graph_status.ok()){
        LOG(ERROR)<<load_graph_status;
        return -1;
    }

    // read image from disk.

    // run session
    std::vector<Tensor> outputs;
    std::vector<std::string> input_names={"input"};
    std::vector<Tensor> input_tensors={Tensor()};
    std::vector<std::string> output_names={"output"};

    Status run_status = session->Run({{input_names[0], input_tensors[0]}},
            output_names, {}, &outputs);
    if(!run_status.ok()){
        LOG(ERROR)<<"Running model failed: "<<run_status;
        return -1;
    }
    // handle output tensors
    return 0;
}
