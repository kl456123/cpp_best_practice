#ifndef CORE_TENSOR_H_
#define CORE_TENSOR_H_
#include <vector>
#include <memory>
#include <CL/cl.hpp>
#include "core/backend.h"


class TensorShape;
class TensorBuffer;


class Tensor{
    struct TensorInfo{
        int dims;
    };


    public:
    class Filler{
        public:
            static void FillTensorByVal(Tensor* tensor,  float value){
                auto data_type = tensor->mDataType;
                for(int i=0;i<tensor->mSize;i++){
                    switch(data_type){
                        case Tensor::DataType::FLOAT32:
                            tensor->host<float>()[i] = float(value);
                            break;
                        case Tensor::DataType::INT32:
                            tensor->host<int32_t>()[i] = static_cast<int32_t>(value);
                            break;
                        case Tensor::DataType::DOUBLE:
                            tensor->host<double>()[i] = static_cast<double>(value);
                            break;
                        case Tensor::DataType::INT8:
                            tensor->host<int8_t>()[i] = static_cast<int8_t>(value);
                            break;
                        default:
                            break;
                    }
                }
            }

            static void FillTensorRandomly(Tensor* tensor){
                float MAX_VALUE=1000.0;
                auto data_type = tensor->mDataType;
                for(int i=0;i<tensor->mSize;i++){
                    switch(data_type){
                        case Tensor::DataType::FLOAT32:
                            tensor->host<float>()[i] = (rand()%int(MAX_VALUE)+1)/static_cast<float>(MAX_VALUE);
                            break;
                        case Tensor::DataType::INT32:
                            tensor->host<int32_t>()[i] = (rand()%int(MAX_VALUE)+1)/static_cast<int32_t>(MAX_VALUE);
                            break;
                        case Tensor::DataType::DOUBLE:
                            tensor->host<double>()[i] = (rand()%int(MAX_VALUE)+1)/static_cast<double>(MAX_VALUE);
                            break;
                        case Tensor::DataType::INT8:
                            tensor->host<int8_t>()[i] = (rand()%int(MAX_VALUE)+1)/static_cast<int8_t>(MAX_VALUE);
                            break;
                        default:
                            break;
                    }
                }
            }

            static void FillTensorGaussian(Tensor* tensor){

            }
    };
    enum DataType{
        INT8=0,
        INT32,
        FLOAT32,
        DOUBLE
    };
    Tensor(const std::vector<int>& tensor_shape, Tensor::DataType data_type, bool alloc);
    Tensor(const std::vector<int>& tensor_shape, Tensor::DataType data_type, void* user_data);
    Tensor(const std::vector<int>& tensor_shape, Tensor::DataType data_type, Backend* backend);
    virtual ~Tensor();


    // static Tensor* Create(const std::vector<int>&tensor_shape, Tensor::DataType data_type);

    static Tensor* Zeros(const std::vector<int>& tensor_shape,Tensor::DataType data_type=Tensor::DataType::FLOAT32);

    static Tensor* Ones(const std::vector<int>& tensor_shape,Tensor::DataType data_type=Tensor::DataType::FLOAT32);

    static Tensor* Random(const std::vector<int>& tensor_shape,Tensor::DataType data_type=Tensor::DataType::FLOAT32);

    static int ComputeSize(const std::vector<int>& shape){
        int size=1;
        for(int i=0;i<shape.size();i++){
            size*=shape[i];
        }
        return size;
    }

    void CopyToDevice(Backend::ForwardType type_name){
        Backend* backend = ExtractBackend(type_name);
        backend->CopyFromHostToDevice(this);
    }


    inline int dims(){
        return mShape.size();
    }

    template<typename T>
        inline T* host(){return (T*)mHost;}

    inline void* host(){
        return mHost;
    }
    inline int size(){
        return mSize;
    }

    const std::vector<int>& shape(){
        return mShape;
    }

    inline size_t buffer_size(){
        return mBufferSize;
    }

    void set_host(void* ptr){
        mHost = ptr;
    }

    void set_device(void* ptr){
        mDevice = ptr;
    }

    void* device(){
        return mDevice;
    }

    DataType type(){
        return mDataType;
    }

    // template<typename T>
    // inline T* device(){return (T*)mDevice;}

    private:
    void* mHost;
    void* mDevice;
    std::vector<int> mShape;
    int mSize;
    size_t mBufferSize;
    bool mOwnMemory;
    DataType mDataType;

};

#endif
