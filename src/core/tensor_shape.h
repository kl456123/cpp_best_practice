#ifndef CORE_TENSOR_SHAPE_H_
#define CORE_TENSOR_SHAPE_H_
#include <string>
#include <limits>
#include <array>
#include <vector>

#include "core/logging.h"
#include "types.pb.h"

// forward declaration
//
template<typename Shape>
class TensorShapeIter;
class TensorShape;
class TensorShapeProto;
class PartialTensorShape;


class TensorShapeRep{
    public:
        virtual ~TensorShapeRep();

        // copy and assign
        TensorShapeRep(const TensorShapeRep&);
        void operator=(const TensorShapeRep&);

        // move
        TensorShapeRep(const TensorShapeRep&&);
        void operator=(const TensorShapeRep&&);

        void Clear();

        static constexpr int MaxDimensions(){return 254;}

        int64_t num_elements(){return num_elements_;}

        // error message
        std::string DebugString()const;
        static std::string DebugString(const TensorShapeProto& proto);

    protected:
        // constructable only via child class
        TensorShapeRep()=default;

        struct Rep16{
            uint16_t dims_[6];
        };

        struct Rep32{
            uint32_t dims_[3];
        };

        struct Rep64{
            std::array<int64_t, 4>* dims_;
        };
        static const int64_t kMaxRep16 = std::numeric_limits<uint16_t>::max()-1;
        static const int64_t kMaxRep32 = std::numeric_limits<uint32_t>::max()-1;
        static const uint16_t kUnknownRep16 = std::numeric_limits<uint16_t>::max();
        static const uint32_t kUnknownRep32 = std::numeric_limits<uint32_t>::max();

        Rep16* as16(){return reinterpret_cast<Rep16*>(buf());}
        Rep32* as32(){return reinterpret_cast<Rep32*>(buf());}
        Rep64* as64(){return reinterpret_cast<Rep64*>(buf());}

        const Rep16* as16()const {return reinterpret_cast<const Rep16*>(buf());}
        const Rep32* as32()const {return reinterpret_cast<const Rep32*>(buf());}
        const Rep64* as64()const {return reinterpret_cast<const Rep64*>(buf());}

        enum class RegTag{REP16=0, REP32, REP_OUT_OF_LINE};

        // data_type
        DataType data_type()const {return static_cast<DataType>(buf()[13]);}
        void set_data_type(DataType dt){
            CHECK_LT(static_cast<uint32_t>(dt), 256u);
            buf()[13] = static_cast<uint8_t>(dt);
        }
        static const uint8_t kUnknownRank = 255;
        // set dim in 14th
        uint8_t ndims_byte()const {return buf()[14];}
        void set_ndims_byte(uint8_t nd){buf()[14]=nd;}

        // set tag in 15th
        void set_tag(RepTag tag){buf()[15] = static_cast<uint8_t>(tag);}
        RegTag tag()const {return static_cast<RegTag>(buf()[15]);}

        void set_num_elements(int64_t n){num_elements_=n;}
    private:
        void DestructorOutOfLine();
        uint8_t* buf(){return &u_.buf[0];}
        const uint8_t* buf()const{return &u_.buf[0];}
        union {
            uint8_t buf[16];
            Rep64* unused_aligner;
        } u_;
        int64_t num_elements_;
};


template<typename Shape>
class TensorShapeBase: public TensorShapeRep{
    public:
        explicit TensorShapeBase(std::vector<int64_t> dim_sizes);
        virtual ~TensorShapeBase();

        TensorShapeBase();
        TensorShapeBase(conset TensorShapeProto& proto);

        static Status IsValidShape(const TensorShapeProto& proto);

        bool IsValid();

        // some operators for tensor shape
        void AddDim(int64_t size);

        void AppendShape(const TensorShapeBase& shape);

        void InsertDim(int d, int64_t size);

        void set_dim(int d, int64_t size);

        void RemoveDim(int d){
            CHECK_GE(d, 0);
            RemoveDimRange(d, d+1);
        }

        void RemoveDimRange(int begin, int end);

        bool unknown_rank()const{
            return ndims_byte()==kUnknownRank;
        }

        int dims()const{
            uint8_t dims = ndims_byte();
            return dims;
        }

        void RemoveLastDims(int n);

        void AsProto(TensorShapeProto* proto)const;

        int64_t dim_size(int d)const;
        std::vector<int64_t> dim_sizes()const;

        // shape iter iterator based loop
        TensorShapeIter<Shape> begin()const;
        TensorShapeIter<Shape> end()const;
    protected:
        explicit TensorShapeBase(DataType dt);
    private:
        void InitDims(std::vector<int64_t> dim_sizes);

};

template<typename Shape>
std::ostream& operator<<(std::ostream& os, const TensorShapeBase<Shape>& tsb){
    return os<<tsb.DebugString();
}


// tensor shape
class TensorShape:public TensorShapeBase<TensorShape>(){
    public:
        using TensorShapeBase<TensorShape>::TensorShapeBase;

        // check the same
        bool IsSameSize(const TensorShape& )const;
        bool operator==(const TensorShape& b)const{return IsSameSize(b);}
        bool operator!=(const TensorShape& b)const{return !IsSameSize(b);}
    private:
        void CheckDimsEqual(int NDIMS)const;
        void CheckDimsAtLeast(int NDIMS)const;
};

// tensorshapdim and shape iter
struct TensorShapeDim{
    explicit TensorShapeDim(int64_t s):size(s){};
    int64_t size;
};

template<typename Shape>
class TensorShapeIter{
    public:
        TensorShape(const Shape* shape, int d):shape_(shape),d_(d){}
        bool operator==(const TensorShapeIter& rhs){
            CHECK(shape_==rhs.shape_);
            return d_==rhs.d_;
        }
        bool operator!=(const TensorShapeIter& rhs){
            CHECK(shape_==rhs.shape_);
            return d_!=rhs.d_;
        }
        void operator++(){++d_;}

        TensorShapeDim operator*(){return TensorShapeDim(shape_->dim_size(d_));}
    private:
        const Shape* shape_;
        int d_;
};


// copy constructor
inline TensorShapeRep::TensorShapeRep(const TensorShapeRep& b){
    num_elements_= b.num_elements_;
    if(b.tag()!=REP_OUT_OF_LINE){
        memcpy(buf(), b.buf(), sizeof(u_.buf));
    }else{
        set_tag(REP16);
        SlowCopyFrom(b);
    }
}

inline TensorShapeRep::TensorShapeRep(TensorShapeRep&& b){
    num_elements_=b.num_elements_;
    memcpy(buf(), b.buf(), sizeof(u_.buf));
    b.set_tag(REP16);
}

inline TensorShapeRep::~TensorShapeRep(){
    if(tag()==REP_OUT_OF_LINE){
        DestructorOutOfLine();
    }
}

inline void TensorShapeRep::operator=(const TensorShapeRep& b){
    num_elements_=b.num_elements_;
    if(tag()!=REP_OUT_OF_LINE&&b.tag()!=REP_OUT_OF_LINE){
        memcpy(buf(), b.buf(), sizeof(u_.buf));
    }else{
        SlowCopyFrom(b);
    }
}

// move assignment
inline void TensorShapeRep::operator=(TensorShapeRep&& b){
    if(tag()==REP_OUT_OF_LINE){
        DestructorOutOfLine();
    }
    num_elements_=b.num_elements_;
    memcpy(buf(), b.buf(), sizeof(u_.buf));
    b.set_tag(REP16);
}

template<typename Shape>
inline TensorShapeBase<Shape>::TensorShapeBase(DataType dt){
    set_tag(REP16);
    set_data_type(dt);

    set_ndims_byte(1);
    uint16_t* dst = as16()->dims_;
    *dst = 0;
    set_num_elements(0);
}

#endif
