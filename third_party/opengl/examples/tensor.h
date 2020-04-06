#ifndef TENSOR_H_
#define TENSOR_H_
#include <vector>

#include "buffer.h"


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
    private:
        std::vector<int> dims_;
};

// device and host sperate storage
class Tensor{
    public:
        Tensor();
        ~Tensor();

        Buffer* device()const{return device_;}
        void* host()const{return host_;}
        bool is_host()const{return host_==nullptr? false: true;}

        size_t num_elements()const{return shape_.num_elements();}


    private:
        Buffer* device_;
        void* host_;

        TensorShape shape_;

        // disallow copy and assign
        Tensor(Tensor& other)=delete;
        Tensor(Tensor&& other)=delete;
        Tensor& operator=(Tensor& other)=delete;
        Tensor& operator=(Tensor&& other)=delete;
};


#endif
