#ifndef TENSOR_H_
#define TENSOR_H_
#include <vector>

#include "opengl/core/texture.h"
#include "opengl/core/buffer.h"
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

            Tensor(DataType dtype, INTLIST shape, MemoryType mem_type);
            template<typename T>
                Tensor(T* data, DataType dtype, INTLIST shape);
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


        private:
            void* device_=nullptr;
            void* host_=nullptr;
            int size_;

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
            CHECK_LE(shapes.size(), 3)<<"Only 1D or 2D input are supported now!";
            CHECK_NE(shapes.size(), 0)<<"Empty tensor is not supported now!";

            if(shapes.size()>=2){
                CHECK_EQ(mem_type, DEVICE_TEXTURE);
            }

            if(shapes.size()==3){
                CHECK_EQ(shapes[2], 4)<<"Only 4 channels mode supported now!";
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
                int tex_h = shapes[0];
                int tex_w = shapes[1];
                device_ = new Texture({tex_h, tex_w}, GL_RGBA32F, GL_TEXTURE_2D, nullptr);
            }else{
                LOG(FATAL)<<"unsupported types!";
            }
        }

}//namespace opengl
#endif
