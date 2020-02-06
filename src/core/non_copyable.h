#ifndef CORE_NON_COPYABLE_H_
#define CORE_NON_COPYABLE_H_


class NonCopyable{
    public:
        NonCopyable()                    = default;
        // copy constructor
        NonCopyable(const NonCopyable&)  = delete;
        NonCopyable(const NonCopyable&&) = delete;
        // assign
        NonCopyable& operator=(const NonCopyable&) = delete;
        NonCopyable& operator=(const NonCopyable&&) = delete;
};


#endif
