#ifndef CORE_DLCL_TENSOR_H_
#define CORE_DLCL_TENSOR_H_
#include <cstddef>
#include "types.pb.h"
#include "core/non_copyable.h"


// forward declarations
class TensorBuffer;
class DLCLTensor;
class TensorShape;


// buffer
class TensorBuffer :public NonCopyable{
    public:
        explicit TensorBuffer(void* data_ptr):data_(data_ptr){}
        ~TensorBuffer(){}

        void* data()const {return data_;}

        virtual size_t size()const=0;

        template<typename T>
            T* base()const{
                return reinterpret_cast<T*>(data());
            }

        virtual bool OwnsMemory()const {return true;}

    private:
        void* const data_;
};

class DLCLTensor{
    public:
        DLCLTensor();
        DLCLTensor(DataType type, const TensorShape& shape);
        DLCLTensor(Allocator* a, DataType type const TensorShape& shape);
        DLCLTensor(DataType type, const TensorShape& shape, TensorBuffer* buf);
        explicit Tensor(DataType type);

        ~Tensor();

        // accessor
        DataType dtype()const {return shape_.data_type();}

        const TensorShape& shape()const {return shape_;}

        int dims()const {return shape().dims();}
        int64_t dim_size(int d) const { return shape().dim_size(d); }
        int64_t NumElements() const { return shape().num_elements(); }

        bool IsSameSize(const Tensor& b)const{
            return shape().IsSameSize(b.shape());
        }

        bool IsInitialized()const;

        size_t TotalBytes()const;

        size_t AllocatedBytes()const;

        bool IsAligned()const {
            return true;
        }


        bool CopyFrom(const Tensor& other, const TensorShape& shape){
            if(other.NumElements()!=shape.num_elements())reutrn false;
            CopyFromInternal(other, shape);
            return ture;
        }
    private:
        TensorShape shape_;
        TensorBuffer* buf_;
};



#endif
