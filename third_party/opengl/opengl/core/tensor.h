#ifndef TENSOR_H_
#define TENSOR_H_
#include <vector>

#include "opengl/core/texture.h"
#include "opengl/core/buffer.h"
#include "opengl/utils/macros.h"
#include "opengl/core/dlxnet.pb.h"
#include <glog/logging.h>

namespace opengl{
    typedef std::vector<int> INTLIST;
    typedef dlxnet::TensorProto::DataFormat DataFormat;

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

            // initialize tensor with shape and type
            // other than content value
            Tensor(DataType dtype, INTLIST shape, MemoryType mem_type=HOST_MEMORY);

            // make tensor from cpu host memory
            template<typename T>
                Tensor(DataType dtype, INTLIST shape, T* data);

            // make tensor from proto
            Tensor(const dlxnet::TensorProto& tensor_proto);
            ~Tensor();

            // get data pointer from device or cpu host
            // note that we just use one of them instead of both
            void* device()const{return device_;}
            void* host()const{return host_;}

            // typed pointer
            template<typename T>
                T* device()const{
                    return reinterpret_cast<T*>(device_);
                }
            template<typename T>
                T* host()const{
                    return reinterpret_cast<T*>(host_);
                }



            // accessor
            template<typename T>
                GLuint device_id(){
                    return reinterpret_cast<T*>(device_)->id();
                }
            const INTLIST& shape()const{return shape_.dims();}
            size_t num_elements()const{return shape_.num_elements();}
            const int size()const{return size_;}
            MemoryType mem_type()const{
                return mem_type_;
            }
            const DataFormat dformat()const{return dformat_;}
            void set_host(void* data){
                host_ = data;
            }

            void set_size(size_t size){
                size_ = size;
            }
            void set_dformat(DataFormat dformat){
                dformat_ = dformat;
            }

            // helper functions
            // TODO(breakpoint) change it member function name
            bool is_host()const{return host_==nullptr? false: true;}
            const bool Initialized()const{
                return initialized_;
            }
            const int channel()const{
                CHECK_EQ(shape_.dims_size(), 4);
                return shape_[3];
            }
            const int width()const{
                CHECK_EQ(shape_.dims_size(), 4);
                return shape_[2];
            }
            const int height()const{
                CHECK_EQ(shape_.dims_size(), 4);
                return shape_[1];
            }
            const int num()const{
                CHECK_EQ(shape_.dims_size(), 4);
                return shape_[0];
            }

        private:
            // inner data pointer
            void* device_=nullptr;
            void* host_=nullptr;

            // help to tell if it is empty or not
            bool initialized_=false;

            //TODO(breakpoint) change it to bytes, more readable
            int size_;

            // common attributes for tensor, like data type, shape and mem type
            DataType dtype_;
            MemoryType mem_type_;
            TensorShape shape_;
            DataFormat dformat_;

            // disallow copy and assign
            Tensor(Tensor& other)=delete;
            Tensor(Tensor&& other)=delete;
            Tensor& operator=(Tensor& other)=delete;
            Tensor& operator=(Tensor&& other)=delete;
    };

    inline Tensor::Tensor(DataType dtype, INTLIST shapes, MemoryType mem_type)
        :shape_(shapes), dtype_(dtype),mem_type_(mem_type){
            // make sure the length of shape equals to 4
            int dims_size = shape_.dims_size();
            if(dims_size<4){
                for(int i=0;i<4-dims_size;++i){
                    shape_.insert_dim(0, 1);
                }
            }
            size_t num_elements, bytes;
            dformat_=dlxnet::TensorProto::NHWC;
            num_elements = shape_.num_elements();

            if(dtype==DT_FLOAT){
                bytes = sizeof(float)* num_elements;
            }else{
                bytes = sizeof(int)*num_elements;
            }


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
                dformat_ = dlxnet::TensorProto::NHWC4;
                device_ = new Texture({image_height, image_width}, GL_RGBA32F, GL_TEXTURE_2D, nullptr);
            }else{
                LOG(FATAL)<<"unsupported types!";
            }

            initialized_=true;
        }

}//namespace opengl
#endif
