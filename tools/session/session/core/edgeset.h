#ifndef SESSION_CORE_EDGESET_H_
#define SESSION_CORE_EDGESET_H_
#include <cstddef>

#include "session/utils/macros.h"

class EdgeSet{
    public:
        EdgeSet();
        virtual ~EdgeSet();

        bool empty()const;

        void clear();

        size_t size()const;
    private:
        static constexpr int kInline = 64/sizeof(const void*);
        const void* ptrs_[kInline];
        DISALLOW_COPY_AND_ASSIGN(EdgeSet);
};

#endif
