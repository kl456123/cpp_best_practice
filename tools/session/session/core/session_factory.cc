#include <unordered_map>

#include "session/core/session_factory.h"
#include "session/core/session_options.h"
#include "session/utils/errors.h"
#include "session/utils/logging.h"

namespace{
    typedef std::unordered_map<std::string, SessionFactory*> SessionFactories;
    SessionFactories* session_factories(){
        static SessionFactories*  factories = new SessionFactories;
        return factories;
    }
}



void SessionFactory::Register(const string& runtime_type,
                              SessionFactory* factory) {
  if (!session_factories()->insert({runtime_type, factory}).second) {
    LOG(ERROR) << "Two session factories are being registered "
               << "under" << runtime_type;
  }
}

Status SessionFactory::GetFactory(const SessionOptions& options,
                                  SessionFactory** out_factory) {
  std::vector<std::pair<string, SessionFactory*>> candidate_factories;
  for (const auto& session_factory : *session_factories()) {
    if (session_factory.second->AcceptsOptions(options)) {
      LOG(INFO) << "SessionFactory type " << session_factory.first
              << " accepts target: " << options.target;
      candidate_factories.push_back(session_factory);
    } else {
      LOG(INFO) << "SessionFactory type " << session_factory.first
              << " does not accept target: " << options.target;
    }
  }

  if (candidate_factories.size() == 1) {
    *out_factory = candidate_factories[0].second;
    return Status::OK();
  } else if (candidate_factories.size() > 1) {
    // NOTE(mrry): This implementation assumes that the domains (in
    // terms of acceptable SessionOptions) of the registered
    // SessionFactory implementations do not overlap. This is fine for
    // now, but we may need an additional way of distinguishing
    // different runtimes (such as an additional session option) if
    // the number of sessions grows.
    // TODO(mrry): Consider providing a system-default fallback option
    // in this case.
    std::vector<string> factory_types;
    factory_types.reserve(candidate_factories.size());
    for (const auto& candidate_factory : candidate_factories) {
      factory_types.push_back(candidate_factory.first);
    }
    return errors::Internal(
        "Multiple session factories registered for the given session options");
  } else {
    return errors::NotFound(
        "No session factory registered for the given session options");
  }
}
