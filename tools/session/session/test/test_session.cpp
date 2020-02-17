#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "session/utils/status.h"
#include "session/core/session.h"
#include "session/core/session_options.h"
#include "session/utils/logging.h"

Status ReadBinaryProto(const string& fname, ::google::protobuf::MessageLite* proto) {
    // std::unique_ptr<RandomAccessFile> file;
    // TF_RETURN_IF_ERROR(env->NewRandomAccessFile(fname, &file));
    // std::unique_ptr<FileStream> stream(new FileStream(file.get()));
    std::fstream input(fname, std::ios::in |std::ios::binary);
    if(!proto->ParseFromIstream(&input)){
        return errors::DataLoss("Can't parse ", fname, " as binary proto");
    }

    // ::google::protobuf::io::CodedInputStream coded_stream(stream.get());
    // Total bytes hard limit / warning limit are set to 1GB and 512MB
    // respectively.
    // coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);

    // if (!proto->ParseFromCodedStream(&coded_stream) ||
    // !coded_stream.ConsumedEntireMessage()) {
    // TF_RETURN_IF_ERROR(stream->status());
    // return errors::DataLoss("Can't parse ", fname, " as binary proto");
    // }
    return Status::OK();
}

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
    RETURN_IF_ERROR(ReadBinaryProto(graph_file_name, &graph_def));

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
    std::string graph_file_name("/home/breakpoint/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb");
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
