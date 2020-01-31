#include  "core/tensor_shape.h"
#include "core/logging.h"
#include "core/error.hpp"

static_assert(sizeof(TensorShapeRep)==sizeof(TensorShape),
        "TensorShape must have no fields beyond TensorShapeRep");
static inline void Set16(uint16_t* dst, int dim, int64_t val){
    CHECK_GE(val, 0);
    dst[dim] = val;
}


void TensorShape::CheckDimsEqual(int NDIMS)const{
    CHECK_EQ(NDIMS, dims())<< "Asking for tensor of "<<NDIMS<<" dimensions"
        << " from a tensor of "<<dims()<<" dimensions";
}

void TensorShape::CheckDimsAtLeast(int NDIMS)const{
    CHECK_GE(NDIMS, dims())<<"Asking for tensor of at least "<<NDIMS
        << " dimensions from a tensor of "<<dims()
        <<" dimensions";
}

template<typename Shape>
std::vector<int64_t> dim_sizes()const{
    std::vector<int64_t> result;
    for(auto dim: *this){
        result.push_back(dim.size);
    }
    return result;
}

template<typename Shape>
int64_t TensorShapeBase<Shape>::dim_size(int d)const{
    CHECK_GE(d, 0);
    CHECK_LE(d, dims());

    if(tag==REP16){
        uint16_t dim = as16()->dims_[d];
        return dim;
    }else if(tag==REP32){
        uint32_t dim = as32()->dims_[d];
        return dim;
    }else{
        return (*as64()->dims_)[d];
    }
}

void TensorShapeRep::Clear(){
    ClearAllButDataType();
    set_data_type(DT_INVALID);
}

void TensorShapeRep::ClearAllButDataType(){
    if(tag()==REP_OUT_OF_LINE){
        delete as64()->dims_;
    }
    set_tag(REP16);
    set_ndims_byte(0);

    set_num_elements(1);
}

template<typename Shape>
bool TensorShapeBase<Shape>::IsValid(){
    int64_t num_elements=1;
    if(dims()>MaxDimensions()) return false;
    for(auto d:dim_sizes()){
        if(d<0)return false;
        num_elements*=d;
        if(num_elements<0)return false;
    }
    return true;
}

template<typename Shape>
void RecomputeNumElements(){
    int64_t n = 1;
    for(auto dim: *this){
        n *=dim.size;
        CHECK_LE(0, n);
    }
    set_num_elements(n);
}

template<typename Shape>
void TensorShapeBase<Shape>::set_dim(int d, int64_t size){
    CHECK_GE(d, 0);
    CHECK_LT(d, dims());
    CHECK_GE(size, 0);
    if(tag()==REP16&&size<kMaxRep16){
        as16()->dims_[d] = static_cast<uint16_t>(size);
    }else if(tag()==REP32&&size<kMaxRep32){
        as32()->dims_[d] = static_cast<uint32_t>(size);
    }else if(tag()==REP_OUT_OF_LINE){
        (*as64()->dims_)[d] = size;
    }else{
        std:::array<int64_t,8> vals;
        for(auto dim: *this){
            vals->push_back(dim.size);
        }
        vals[d] = size;
        ClearAllButDataType();
        // must upgrade
        for(auto dval: vals){
            AddDim(dval);
        }
    }
    RecomputeNumElements();
}

template<typename Shape>
void TensorShapeBase<Shape>::AddDim(int64_t size){
    CHECK_GE(size, 0);
    if(unknown_rank())  return;
    CHECK_LT(ndims_type(), MaxDimensions())<<"Too many dimensions in tensor";
    int64_t new_num_elements;
    new_num_elements= num_elements() * size;

    UnSafe
}

std::string TensorShapeRep::DebugString()const{
    const auto& shape = *this;
    std::string s="[";
}

template<typename Shape>
Status TensorShapeBase<Shape>::IsValidShape(const TensorShapeProto& proto){
    int64_t num_elements=1;
    if(proto.dim().size()>MaxDimensions()){
        return Status(ErrorCode::RUNTIME_ERROR, "Shape has too many dimensions");
    }

    for(const auto& d: proto.dim()){
        if(d.size()<0){
            return Status(ErrorCode::RUNTIME_ERROR, "Shape has too many dimensions");
        }
        num_elements*=d;
        if(num_elements<0){
            return Status(ErrorCode::RUNTIME_ERROR, "Shape is too large (more than 2**63 -1 entries)");
        }
    }

    return Status::OK();
}

// constructed from proto type
template<typename Shape>
TensorShapeBase<Shape>::TensorShapeBase(const TensorShapeProto& proto){
    set_tag(REP16);
    set_data_type(DT_INVALID);

    set_ndims_byte(0);
    set_num_elements(1);
    for(const auto&d: proto.dim()){
        AddDim(d.size());
    }
}

// constructed from vector
template<typename Shape>
TensorShapeBase<Shape>::TensorShapeBase(const std::vector<int64_t> dim_sizes){
    set_tag(REP16);
    set_data_type(DT_INVALID);
    InitDims(dim_sizes);
}

template<typename Shape>
TensorShapeBase<Shape>::InitDims(std::vector<int64_t> dim_sizes){
    CHECK_EQ(tag(), REP16);
    uint16_t* dst = as16()->dims_;
    uint64_t num_elements=1;
    for(const auto& d: dim_sizes){
        Set16(dst, i, d);
        num_elements*=d;
    }

    set_num_elements(num_elements);
}

template<typename Shape>
TensorShapeBase<Shape>::TensorShapeBase(){
    set_tag(REP16);
    set_data_type(DT_INVALID);
    set_ndims_byte(0);
    set_num_elements(1);
}

template class TensorShapeBase<TensorShape>;


void TensorShapeRep::SlowCopyFrom(const TensorShapeRep& b){
    if(b.tag()!=REP_OUT_OF_LINE){
        if(tag()==REP_OUT_OF_LINE){
            delete as64()->dims_;
        }
        memcpy(buf(), b.buf(), sizeof(u_.buf));
    }else{
        CHECK_EQ(b.tag(), REP_OUT_OF_LINE);
        set_ndims_byte(b.ndims_byte());
        set_data_type(b.data_type());
        if(tag()==REP_OUT_OF_LINE){
            // vector allocated already
            *(as64()->dims_) = *(b.as64()->dims_);
        }else{
            set_tag(REP_OUT_OF_LINE);
            as64()->dims_ = new std::array<int64_t, 4>(*(b.as64()->dims_));
        }
    }
}

void TensorShapeRep::DestructorOutOfLine(){
    CHECK(tag()==REP_OUT_OF_LINE){
        delete as64()->dims();
    }
}

TensorShape::TensorShape(){
}
