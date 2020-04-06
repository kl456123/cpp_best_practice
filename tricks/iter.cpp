/**
 * How to use iterator
 */
#include <iostream>

enum DataType{
    DT_INT8,
    DT_INT32
};

class DataTypeSet{
    private:
        const uint32_t mask_;
        static constexpr uint32_t kNumBits = 32;

    public:
        constexpr DataTypeSet(const DataTypeSet& other):mask_(other.mask_){}
        explicit constexpr DataTypeSet(uint32_t mask):mask_(mask){}

        constexpr bool Contains(DataType dt)const{
            return (static_cast<uint32_t>(dt)<kNumBits)&&
                ((mask_>>static_cast<uint32_t>(dt))&1u)!=0u;
        }

        // iterator
        class Iterator{
            const DataTypeSet& set_;
            uint32_t pos_;
            public:
            Iterator(const DataTypeSet& set, uint32_t pos):set_(set), pos_(pos){
                // DCHECK_LE(pos, kNumBits);
            }
            DataType operator*()const {return static_cast<DataType>(pos_);}
            Iterator& operator++(){
                ++pos_;
                //skip zero(empty slot) in the middle
                if(pos_<kNumBits){
                    uint32_t remain_mask = set_.mask_>>pos_;
                    if(remain_mask!=0u){
                        pos_+=ctz_uint32(remain_mask);
                    }
                }
                return *this;
            }

            bool operator==(const Iterator& other)const{return pos_==other.pos_;}
            bool operator!=(const Iterator& other)const{return !(*this==other);}
            size_t operator-(const Iterator& other){
                return this->pos_-other.pos_;
            }
        };

        static uint32_t ctz_uint32(uint32_t x) {
#ifdef __GNUC__
            return __builtin_ctz(x);
#else
            // bit in the most bottom is 0
            uint32_t n = 0u;
            while((x&1u)==0u){
                x>>=1;
                n++;
            }
            return n;

#endif
        }

        static uint32_t clz_uint32(uint32_t x) {
#ifdef __GNUC__
            return __builtin_clz(x);
#else
            uint32_t n = 0u;
            // bit in the most top is 0
            while((x&(kNumBits-1u))==0u){
                x<<=1;
                n++;
            }
            return n;
#endif
        }


        Iterator begin()const{
            if(mask_!=0){
                return Iterator(*this, ctz_uint32(mask_));
            }
            return Iterator(*this, 0);
        }

        Iterator end()const{
            // The end position is the index of the highest bit that is set, plus 1.
            // If there are no bits set to 1, then the index is 0.
            if (mask_ != 0) {
                return Iterator(*this, kNumBits - clz_uint32(mask_));
            }
            // The set is empty.
            return Iterator(*this, 0);
        }

        size_t size()const{
#if defined(__GNUC__)
            return __builtin_popcount(mask_);
#else
            size_t n = 0u;
            while(x>0){
                n+=x&1u;
                x>>1;
            }
            return n;
#endif
        }

        constexpr DataTypeSet operator|(const DataTypeSet& other)const{
            return DataTypeSet(mask_|other.mask_);
        }
};

constexpr inline DataTypeSet ToSet(DataType dt){
    return DataTypeSet(1u<<static_cast<uint32_t>(dt));
}

constexpr DataTypeSet kAllTypes = ToSet(DT_INT8)|ToSet(DT_INT32);
inline const DataTypeSet& AllTypes(){return kAllTypes;}

int main(){
    for(DataType dt: AllTypes()){
        std::cout<<dt<<std::endl;
    }
    return 0;
}
