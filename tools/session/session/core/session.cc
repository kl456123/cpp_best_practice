#include "session/core/session.h"
#include "session/core/session_factory.h"
#include "session/utils/logging.h"


Session::Session(){}
Session::~Session(){}

Status Session::Run(const RunOptions& run_options,
        const std::vector<std::pair<string, Tensor> >& inputs,
        const std::vector<string>& output_tensor_names,
        const std::vector<string>& target_node_names,
        std::vector<Tensor>* outputs, RunMetadata* run_metadata) {
    return errors::Unimplemented(
            "Run with options is not supported for this session.");
}


Status NewSession(const SessionOptions& options, Session** out_session) {
    SessionFactory* factory;
    Status s = SessionFactory::GetFactory(options, &factory);
    if (!s.ok()) {
        *out_session = nullptr;
        LOG(ERROR) << s;
        return s;
    }
    // Starts exporting metrics through a platform-specific monitoring API (if
    // provided). For builds using "tensorflow/core/platform/default", this is
    // currently a no-op.
    // session_created->GetCell()->Set(true);
    // monitoring::StartExporter();
    s = factory->NewSession(options, out_session);
    if (!s.ok()) {
        *out_session = nullptr;
    }
    return s;
}
