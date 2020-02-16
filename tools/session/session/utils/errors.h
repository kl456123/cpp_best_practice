#ifndef SESSION_UTILS_ERRORS_H_
#define SESSION_UTILS_ERRORS_H_
#include "error_codes.pb.h"
#include "session/utils/strcat.h"
#include "session/utils/status.h"

namespace errors{
    typedef error::Code Code;

#define DECLARE_ERROR(FUNC, CONST)                                          \
    template<typename... Args>                                              \
    Status FUNC(Args... msg){                                               \
        return Status(error::CONST, string_utils::str_cat(msg...));         \
    }                                                                       \
    inline bool Is##FUNC(const Status& status) {              \
        return status.code() == error::CONST;                 \
    }


DECLARE_ERROR(Cancelled, CANCELLED)
DECLARE_ERROR(InvalidArgument, INVALID_ARGUMENT)
DECLARE_ERROR(NotFound, NOT_FOUND)
DECLARE_ERROR(AlreadyExists, ALREADY_EXISTS)
DECLARE_ERROR(ResourceExhausted, RESOURCE_EXHAUSTED)
DECLARE_ERROR(Unavailable, UNAVAILABLE)
DECLARE_ERROR(FailedPrecondition, FAILED_PRECONDITION)
DECLARE_ERROR(OutOfRange, OUT_OF_RANGE)
DECLARE_ERROR(Unimplemented, UNIMPLEMENTED)
DECLARE_ERROR(Internal, INTERNAL)
DECLARE_ERROR(Aborted, ABORTED)
DECLARE_ERROR(DeadlineExceeded, DEADLINE_EXCEEDED)
DECLARE_ERROR(DataLoss, DATA_LOSS)
DECLARE_ERROR(Unknown, UNKNOWN)
DECLARE_ERROR(PermissionDenied, PERMISSION_DENIED)
DECLARE_ERROR(Unauthenticated, UNAUTHENTICATED)

#undef DECLARE_ERROR

#define RETURN_IF_ERROR(...)                                \
        do{                                                 \
            Status _status = (__VA_ARGS__);                 \
            if(PREDICT_FALSE(!_status.ok()))return _status; \
        }while(0)

}



#endif
