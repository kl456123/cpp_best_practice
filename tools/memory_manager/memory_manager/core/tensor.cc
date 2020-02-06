#include "memory_manager/core/tensor.h"
#include "memory_manager/core/typed_allocator.h"
#include "allocation_description.pb.h"
#include "memory_manager/core/types.h"

bool TensorBuffer::GetAllocatedBytes(size_t* out_bytes) const {
    AllocationDescription allocation_description;
    FillAllocationDescription(&allocation_description);
    if (allocation_description.allocated_bytes() > 0) {
        *out_bytes = allocation_description.allocated_bytes();
        return true;
    } else {
        return false;
    }
}

namespace{
    // some inner classes
    // An un-templated base class for Buffer.
    class BufferBase : public TensorBuffer {
        public:
            explicit BufferBase(Allocator* alloc, void* data_ptr)
                : TensorBuffer(data_ptr), alloc_(alloc) {}

            bool GetAllocatedBytes(size_t* out_bytes) const override {
                if (alloc_->TracksAllocationSizes()) {
                    *out_bytes = alloc_->AllocatedSize(data());
                    return *out_bytes > 0;
                } else {
                    return false;
                }
            }

            void FillAllocationDescription(AllocationDescription* proto) const override {
                void* data_ptr = data();
                int64_t rb = size();
                proto->set_requested_bytes(rb);
                proto->set_allocator_name(alloc_->Name());
                proto->set_ptr(reinterpret_cast<uintptr_t>(data_ptr));
                if (alloc_->TracksAllocationSizes()) {
                    int64_t ab = alloc_->AllocatedSize(data_ptr);
                    proto->set_allocated_bytes(ab);
                    int64_t id = alloc_->AllocationId(data_ptr);
                    if (id > 0) {
                        proto->set_allocation_id(id);
                    }
                }
            }

        protected:

            Allocator* const alloc_;
    };

    // Typed ref-counted buffer: T[n].
    template <typename T>
        class Buffer : public BufferBase {
            public:
                Buffer(Allocator* a, int64_t n);
                Buffer(Allocator* a, int64_t n, const AllocationAttributes& allocation_attr);

                size_t size() const override { return sizeof(T) * elem_; }

            private:
                int64_t elem_;

                ~Buffer() override;

                DISALLOW_COPY_AND_ASSIGN(Buffer);
        };
    // A set of helper functions depending on T.
    template <typename T>
        struct Helper {

            // Memory usage.
            static int64_t TotalBytes(TensorBuffer* in, int64_t n) {
                CHECK_EQ(in->size(), sizeof(T) * n);
                return in->size();
            }
        };

    template <typename T>
        Buffer<T>::Buffer(Allocator* a, int64_t n)
        : BufferBase(a, TypedAllocator::Allocate<T>(a, n, AllocationAttributes())),
        elem_(n) {}

    template <typename T>
        Buffer<T>::Buffer(Allocator* a, int64_t n,
                const AllocationAttributes& allocation_attr)
        : BufferBase(a, TypedAllocator::Allocate<T>(a, n, allocation_attr)),
        elem_(n) {}

    template <typename T>
        Buffer<T>::~Buffer() {
            if (data()) {
                TypedAllocator::Deallocate<T>(alloc_, static_cast<T*>(data()), elem_);
            }
        }
};// end namespace


Tensor::Tensor() : Tensor(DT_FLOAT) {}

Tensor::Tensor(DataType type) : shape_(type), buf_(nullptr) {}

Tensor::Tensor(DataType type, const TensorShape& shape, TensorBuffer* buf)
    : shape_(shape), buf_(buf) {
        set_dtype(type);
    }

bool Tensor::IsInitialized() const {
    return (buf_ != nullptr && buf_->data() != nullptr) ||
        shape_.num_elements() == 0;
}

Tensor::~Tensor() {}
// The macro CASES() expands to a switch statement conditioned on
// TYPE_ENUM. Each case expands the STMTS after a typedef for T.
#define SINGLE_ARG(...) __VA_ARGS__
#define CASE(TYPE, STMTS)             \
    case DataTypeToEnum<TYPE>::value: { \
                                          typedef TYPE T;                   \
                                          STMTS;                            \
                                          break;                            \
                                      }
#define CASES_WITH_DEFAULT(TYPE_ENUM, STMTS, INVALID, DEFAULT) \
    switch (TYPE_ENUM) {                                         \
        CASE(float, SINGLE_ARG(STMTS))                             \
        CASE(double, SINGLE_ARG(STMTS))                            \
        CASE(int32_t, SINGLE_ARG(STMTS))                           \
        case DT_INVALID:                                           \
                                                                   INVALID;                                                 \
        break;                                                   \
        default:                                                   \
                                                                   DEFAULT;                                                 \
        break;                                                   \
    }

#define CASES(TYPE_ENUM, STMTS)                                      \
    CASES_WITH_DEFAULT(TYPE_ENUM, STMTS, LOG(FATAL) << "Type not set"; \
            , LOG(FATAL) << "Unexpected type: " << TYPE_ENUM;)


Tensor::Tensor(Allocator* a, DataType type, const TensorShape& shape)
    : shape_(shape), buf_(nullptr) {
        set_dtype(type);
        // CHECK_NOTNULL(a);
        if (shape_.num_elements() > 0 || a->AllocatesOpaqueHandle()) {
            CASES(type, buf_ = new Buffer<T>(a, shape.num_elements()));
        }
    }

Tensor::Tensor(Allocator* a, DataType type, const TensorShape& shape,
        const AllocationAttributes& allocation_attr)
    : shape_(shape), buf_(nullptr) {
        set_dtype(type);
        // CHECK_NOTNULL(a);
        if (shape_.num_elements() > 0 || a->AllocatesOpaqueHandle()) {
            CASES(type, buf_ = new Buffer<T>(a, shape.num_elements(), allocation_attr));
        }
    }

static Allocator* get_default_cpu_allocator() {
    static Allocator* default_cpu_allocator =
        cpu_allocator();
    return default_cpu_allocator;
}

Tensor::Tensor(DataType type, const TensorShape& shape)
    : Tensor(get_default_cpu_allocator(), type, shape) {}


size_t Tensor::TotalBytes() const {
    if (shape_.num_elements() == 0) return 0;
    CHECK(buf_) << "null buf_ with non-zero shape size " << shape_.num_elements();
    CASES(dtype(), return Helper<T>::TotalBytes(buf_, shape_.num_elements()));
    return 0;  // Makes compiler happy.
}

size_t Tensor::AllocatedBytes() const {
    if (buf_) {
        size_t ret;
        if (buf_->GetAllocatedBytes(&ret)) {
            return ret;
        }
    }
    return TotalBytes();
}

#undef CASES
#undef CASE

