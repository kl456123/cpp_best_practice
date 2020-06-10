#ifndef TENSOR_H_
#define TENSOR_H_
/*      Core Data Class in All DeepLearning Framework
 * class Tensor includes tensor shape and its buffer storing real data
 * and contains some informations like dtype, dformat, mem_type and etc.
 *      As for user, all should to know is that data should only be initilized
 * in host memory, if you want to initialize data whatever you want in device,
 * you must copy data from host to memory by youself, Initialize data in device
 * is not allowed.
 */
#include <vector>

#include "opengl/core/types.h"
#include "opengl/core/texture.h"
#include "opengl/core/buffer.h"
#include "opengl/utils/macros.h"
#include "opengl/core/dlxnet.pb.h"
#include "opengl/core/ogl_allocator.h"
#include <glog/logging.h>

namespace dlxnet{
    class TensorDescription;
}
namespace opengl{
    using dlxnet::TensorDescription;
    //TODO(breakpoint) put all tensor attributes in a struct
    // enum Tensor::DataType;
    // enum Tensor::MemoryType;

    // struct TensorAttributes{
    // Tensor::DataType dtype;
    // Tensor::MemoryType mem_type;
    // Tensor::DataFormat dformat;
    // };

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

            const IntList& dims()const{return dims_;}
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
            Tensor(DataType dtype, IntList shape, MemoryType mem_type=HOST_MEMORY,
                    DataFormat dformat=dlxnet::TensorProto::NHWC);

            // make tensor from cpu host memory, data is defined by user.
            template<typename T>
                Tensor(DataType dtype, IntList shape, T* data,
                        DataFormat dformat=dlxnet::TensorProto::NHWC);

            // make tensor from proto, common used when
            // loading graph proto model
            Tensor(const dlxnet::TensorProto& tensor_proto);

            // to help test or debug, no need to allocate data by user
            // some helper funcs zeros, ones, empty, random
            // note that callee dont own it.
            static Tensor* Random(DataType dtype, IntList shape,
                    DataFormat dformat=dlxnet::TensorProto::NHWC);
            static Tensor* Empty(DataType dtype, IntList shape,
                    DataFormat dformat=dlxnet::TensorProto::NHWC);
            static Tensor* Ones(DataType dtype, IntList shape,
                    DataFormat dformat=dlxnet::TensorProto::NHWC);
            static Tensor* Zeros(DataType dtype, IntList shape,
                    DataFormat dformat=dlxnet::TensorProto::NHWC);
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
            const IntList& shape()const{return shape_.dims();}
            size_t num_elements()const{return shape_.num_elements();}
            const int size()const{return size_;}
            const DataType dtype()const{return dtype_;}
            MemoryType mem_type()const{
                return mem_type_;
            }
            const int dims_size()const{return shape_.dims_size();}
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
                CheckIsNotGeneralTensor();
                const auto dims_size = shape_.dims_size();
                if(dformat_==dlxnet::TensorProto::NHWC
                        ||dformat_==dlxnet::TensorProto::NHWC4){
                    return shape_[dims_size-1];
                }
                if(dims_size>=3){
                    return shape_[dims_size-3];
                }
                return 1;
            }
            const int width()const{
                CheckIsNotGeneralTensor();
                const auto dims_size = shape_.dims_size();
                if(dformat_==dlxnet::TensorProto::NHWC
                        ||dformat_==dlxnet::TensorProto::NHWC4){
                    if(dims_size>=2){
                        return shape_[shape_.dims_size()-2];
                    }
                    return 1;
                }
                return shape_[dims_size-1];
            }

            const int height()const{
                CheckIsNotGeneralTensor();
                const auto dims_size = shape_.dims_size();
                if(dformat_==dlxnet::TensorProto::NHWC
                        ||dformat_==dlxnet::TensorProto::NHWC4){
                    if(dims_size>=3){
                        return shape_[shape_.dims_size()-3];
                    }
                    return 1;
                }
                // nchw or hwn4c4
                // note that for hwn4c4, it is designed for
                // filter. due to filter is nchw(n_out,n_in, h, w)
                // and it is not changed during transformation so it fall
                // to nchw case
                if(dims_size>=2){
                    return shape_[dims_size-2];
                }
                return 1;
            }

            void CheckIsNotGeneralTensor()const{
                CHECK_NE(dformat(), dlxnet::TensorProto::ANY);
                CHECK_NE(dformat(), dlxnet::TensorProto::ANY4);
            }

            const int num()const{
                CheckIsNotGeneralTensor();
                const auto dims_size = shape_.dims_size();
                if(dims_size>=4){
                    return shape_[dims_size-4];
                }
                return 1;

            }

            const int last_stride()const{return shape_[shape_.dims_size()-1];}

            std::string DebugString()const;
            std::string ShortDebugString()const;

            void AsProto(dlxnet::TensorProto* proto)const;

            void CheckShapeAndDFormat(){
                // self-check
                if(shape_.dims_size()!=4){
                    CHECK(dformat()==::dlxnet::TensorProto::ANY
                            ||dformat()==::dlxnet::TensorProto::ANY4);
                }
            }
            void FillDescription(TensorDescription* description)const;

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

    inline Tensor::Tensor(DataType dtype, IntList shapes, MemoryType mem_type, DataFormat dformat)
        :shape_(shapes), dtype_(dtype),mem_type_(mem_type), dformat_(dformat){
            // AmendShape();
            CheckShapeAndDFormat();
            size_t num_elements, bytes;
            num_elements = shape_.num_elements();

            if(dtype==DT_FLOAT){
                bytes = sizeof(float)* num_elements;
            }else{
                bytes = sizeof(int)*num_elements;
            }

            // shape and type
            size_ = bytes;
            if(mem_type==HOST_MEMORY){
                // if(dformat_==dlxnet::TensorProto::ANY4){
                    // host_ = StrideAllocator::Allocate(ogl_texture_allocator(),
                            // size_, last_stride(), AllocationAttributes());
                // }else{
                    host_= new float[num_elements];
                // }
            }else if(mem_type==DEVICE_BUFFER){
                device_ = new ShaderBuffer(bytes);
            }else if(mem_type==DEVICE_TEXTURE){
                int image_height, image_width ;
                if(dformat_==dlxnet::TensorProto::NHWC4){
                    // when use texture, reorganize the shape to (H, W, 4)
                    image_height = num()*height();
                    image_width = UP_DIV(channel(), 4) * width();
                }else if(dformat_==dlxnet::TensorProto::HWN4C4){
                    // by default, shape_ = (N_out, N_in, h, w)
                    // image (H*W*N4, C4*4, 4), merge N4 to height due to
                    // spatial dim is small in most common cases for filter tensor
                    image_height = width()*height()*UP_DIV(num(), 4);
                    image_width = UP_DIV(channel(), 4)*4;
                }else if(dformat==dlxnet::TensorProto::ANY4){
                    // device_ = StrideAllocator::Allocate(ogl_texture_allocator(),
                            // size_, last_stride(), AllocationAttributes());
                    // image_height = device<Texture>()->height();
                    // image_width = device<Texture>()->width();
                    // initialized_=true;
                    // return ;
                    image_height = 1;
                    const int dims = dims_size();
                    const int src_last_dim = shape_[dims-1];
                    const int dst_last_dim = UP_ROUND(src_last_dim, 4);
                    image_width = num_elements / src_last_dim * UP_DIV(src_last_dim, 4);
                }else{
                    LOG(FATAL)<<"unsupported data format: "<<dformat_ <<" for mem_type: "<<mem_type;
                }
                if(!device_){
                    CHECK_GT(image_width, 0);
                    CHECK_GT(image_height, 0);
                    size_ = image_height*image_width*4*sizeof(float);
                    device_ = new Texture({image_width, image_height}, GL_RGBA32F, GL_TEXTURE_2D, nullptr);
                }
            }else{
                LOG(FATAL)<<"unsupported types!";
            }

            initialized_=true;
        }

}//namespace opengl
#endif
