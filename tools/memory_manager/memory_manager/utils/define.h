#ifndef CORE_DEFINE_H_
#define CORE_DEFINE_H_

#define MEMORY_ALIGN_DEFAULT 1<<5
#define DELTA 1e-5


#define DLCL_ASSERT(cond)               \
    if(!cond){                          \
        THROW_ERROR("Assert error!\n"); \
    }

#define PREDICT_FALSE(x)    (__builtin_expect(x, 0))
#define PREDICT_TRUE(x)     (__builtin_expect(!!(x), 1))

#define DISALLOW_COPY_AND_ASSIGN(TypeName)        \
    TypeName(const TypeName&)=delete;      \
    void operator=(const TypeName&)=delete

#define DLCL_MUST_USE_RESULT __attribute__((warn_unused_result))


#endif
