#ifdef SELECTIVE_REGISTRATION
// define SHOULD_REGISTER for each op to selective register them
#else
#define SHOULD_REGISTER_OP(op) true
#define SHOULD_REGISTER_GRADIENT(op) true
#define SHOULD_REGISTER_KERNEL(op) true
#endif
