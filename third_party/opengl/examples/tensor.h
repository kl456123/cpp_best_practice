#ifndef TENSOR_H_
#define TENSOR_H_
#include <vector>

#include "texture.h"

typedef std::vector<int> INTLIST;


class TensorShape{
    public:
        TensorShape(){
        }
        size_t num_elements()const{
            size_t size = 1;
            for(auto dim:dims_){
                size*=dim;
            }
            return size;
        }

        const INTLIST& dims()const{return dims_;}
    private:
        std::vector<int> dims_;
};

// device and host sperate storage
class Tensor{

    public:
        enum DataType{
            DT_INT=0,
            DT_FLOAT=1,
            DT_DOUBLE=2,
            DT_INVALID
        };
        Tensor();
        template<typename T>
            Tensor(T* data, DataType dtype);
        ~Tensor();

        void* device()const{return device_;}
        void* host()const{return host_;}
        bool is_host()const{return host_==nullptr? false: true;}

        size_t num_elements()const{return shape_.num_elements();}
        const INTLIST& dims(){return shape_.dims();}

        template<typename T>
        GLuint device_id(){
            return reinterpret_cast<T*>(device_)->id();
        }


    private:
        void* device_;
        void* host_;

        DataType dtype_;

        TensorShape shape_;

        // disallow copy and assign
        Tensor(Tensor& other)=delete;
        Tensor(Tensor&& other)=delete;
        Tensor& operator=(Tensor& other)=delete;
        Tensor& operator=(Tensor&& other)=delete;
};

inline Tensor::Tensor(){
    host_=nullptr;
    dtype_=DT_INVALID;
}



#endif
