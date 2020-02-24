#ifndef STREAM_EXECUTOR_UTILS_STATUS_MACROS_H_
#define STREAM_EXECUTOR_UTILS_STATUS_MACROS_H_

#define SE_RETURN_IF_ERROR(__status)        \
    do{                                     \
        auto status = __status;             \
        if(!status.ok()){                   \
            return status;                  \
        }                                   \
    }while(false)


#define SE_MACRO_CONCAT(__x, __y)         SE_MACRO_CONCAT_INNER(__x, __y)
#define SE_MACRO_CONCAT_INNER(__x, __y)    __x##__y

#define SE_ASSIGN_OR_RETURN_IMPL(__lhs, __rhs, __name)      \
    auto __name = (__rhs);                                  \
    if(!__name.ok()){                                       \
        return __name.status();                             \
    }                                                       \
    __lhs = std::move(__name.ValueOrDie());

#define SE_ASSIGN_OR_RETURN(__lhs, __rhs)                       \
    SE_ASSIGN_OR_RETURN_IMPL(__lhs, __rhs,                      \
            SE_MACRO_CONCAT(__status__or_value, __COUNTER__));


#endif
