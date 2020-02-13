#ifndef SESSION_CORE_SESSION_FACTORY_H_
#define SESSION_CORE_SESSION_FACTORY_H_
#include "session/utils/status.h"


class Session;
struct SessionOptions;

class SessionFactory {
    public:
        // Creates a new session and stores it in *out_session, or fails with an error
        // status if the Session could not be created. Caller takes ownership of
        // *out_session if this returns Status::OK().
        virtual Status NewSession(const SessionOptions& options,
                Session** out_session) = 0;

        virtual bool AcceptsOptions(const SessionOptions& options) = 0;

        virtual ~SessionFactory() {}
        static void Register(const string& runtime_type, SessionFactory* factory);
        static Status GetFactory(const SessionOptions& options,
                SessionFactory** out_factory);
};



#endif
