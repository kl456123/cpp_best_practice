#ifndef TENSOR_H_
#define TENSOR_H_
#include <vector>

#include "buffer_base.h"

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
        Tensor();
        ~Tensor();

        BufferBase* device()const{return device_;}
        void* host()const{return host_;}
        bool is_host()const{return host_==nullptr? false: true;}

        GLuint device_id(){return device_==nullptr? 0: device_->id();}

        size_t num_elements()const{return shape_.num_elements();}
        const INTLIST& dims(){return shape_.dims();}


    private:
        BufferBase* device_;
        void* host_;

        TensorShape shape_;

        // disallow copy and assign
        Tensor(Tensor& other)=delete;
        Tensor(Tensor&& other)=delete;
        Tensor& operator=(Tensor& other)=delete;
        Tensor& operator=(Tensor&& other)=delete;
};


#endif
