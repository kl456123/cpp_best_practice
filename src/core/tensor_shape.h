#ifndef CORE_TENSOR_SHAPE_H_
#define CORE_TENSOR_SHAPE_H_
#include <string>
#include <limits>
#include <array>
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

        void set_data_type(DataType dt){
            buf()[13] = static_cast<uint8_t>(dt);
        }
        void set_tag(){}
        void set_num_elements(int64_t n){num_elements_=n;}
    private:
        uint8_t* buf(){return &u_.buf[0];}
        const uint8_t* buf()const{return &u_.buf[0];}
        union {
            uint8_t buf[16];
            Rep64* unused_aligner;
        } u_;
        int64_t num_elements_;

};


class TensorShape{
    public:
        TensorShape();
        virtual ~TensorShape();
};
#endif
