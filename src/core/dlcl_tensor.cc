#include "core/dlcl_tensor.h"


class Allocator;
namespace {
    // it contains alloc_ and data_
    class BufferBase: public TensorBuffer{
        public:
            explicit BufferBase(Allocator* alloc, void* data_ptr)
                :TensorBuffer(data_ptr), alloc_(alloc){}
        protected:
            Allocator const alloc_;
    };

    template<typename T>
        class Buffer:public BufferBase{
            public:
                Buffer(Allocator* a, int64_t n);
                size_t size()const {return sizeof(T) * elem_;}
            private:
                int64_t elem_;
                ~Buffer()override;
                DISALLOW_COPY_AND_ASSIGN(Buffer);
        };
}


template<typename T>
Buffer<T>::Buffer(Allocator& a, int64_t n)
    :BufferBase(a, TypedAllocator){
}
