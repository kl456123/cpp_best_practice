#ifndef TENSOR_H_
#define TENSOR_H_
#include <vector>

#include "opengl/core/texture.h"
#include "opengl/core/buffer.h"
#include "opengl/utils/macros.h"
#include <glog/logging.h>

namespace opengl{
    typedef std::vector<int> INTLIST;


    class TensorShape{
        public:
            TensorShape(std::vector<int>& dims)
                :dims_(dims){}
            TensorShape(){}
            size_t num_elements()const{
                size_t size = 1;
                for(auto dim:dims_){
                    size*=dim;
                }
                return size;
            }

            const INTLIST& dims()const{return dims_;}
            void add_dim(int dim){
                dims_.emplace_back(dim);
            }
            void insert_dim(int i, int dim){
                // insanity check
                dims_.insert(dims_.begin()+i, dim);
            }

            const int dims_size()const{
                return dims_.size();
            }

            const int operator[](int i)const{
                return dims_[i];
            }
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

            enum MemoryType{
                HOST_MEMORY=0,
                DEVICE_TEXTURE,
                DEVICE_BUFFER
            };

            Tensor(DataType dtype, INTLIST shape, MemoryType mem_type=HOST_MEMORY);
            template<typename T>
                Tensor(DataType dtype, INTLIST shape, T* data);
            ~Tensor();

            void* device()const{return device_;}
            template<typename T>
                T* device()const{
                    return reinterpret_cast<T*>(device_);
                }
            void* host()const{return host_;}
            template<typename T>
                T* host()const{
                    return reinterpret_cast<T*>(host_);
                }
            bool is_host()const{return host_==nullptr? false: true;}

            size_t num_elements()const{return shape_.num_elements();}
            const INTLIST& shape(){return shape_.dims();}

            const int size()const{return size_;}

            template<typename T>
                GLuint device_id(){
                    return reinterpret_cast<T*>(device_)->id();
                }

            MemoryType mem_type()const{
                return mem_type_;
            }

            const bool Initialized()const{
                return initialized_;
            }


        private:
            void* device_=nullptr;
            void* host_=nullptr;
            int size_;
            bool initialized_=false;

            DataType dtype_;
            MemoryType mem_type_;

            TensorShape shape_;

            // disallow copy and assign
            Tensor(Tensor& other)=delete;
            Tensor(Tensor&& other)=delete;
            Tensor& operator=(Tensor& other)=delete;
            Tensor& operator=(Tensor&& other)=delete;
    };

    inline Tensor::Tensor(DataType dtype, INTLIST shapes, MemoryType mem_type)
        :shape_(shapes), dtype_(dtype),mem_type_(mem_type){
            // make sure the length of shape equals to 4
            if(shape_.dims_size()<4){
                for(int i=0;i<4-shape_.dims_size();++i){
                    shape_.insert_dim(0, 1);
                }
            }

            size_t num_elements = shape_.num_elements();
            const size_t bytes = sizeof(float)* num_elements;

            // shape and type
            size_ = bytes;
            if(mem_type==HOST_MEMORY){
                host_= new float[num_elements];
            }else if(mem_type==DEVICE_BUFFER){
                device_ = new ShaderBuffer(bytes);
            }else if(mem_type==DEVICE_TEXTURE){
                // when use texture, reorganize the shape to (H, W, 4)
                const int image_height = shape_[0]*shape_[1];
                const int image_width = UP_DIV(shape_[3], 4) * shape_[2];
                device_ = new Texture({image_height, image_width}, GL_RGBA32F, GL_TEXTURE_2D, nullptr);
            }else{
                LOG(FATAL)<<"unsupported types!";
            }

            initialized_=true;
        }

}//namespace opengl
#endif
