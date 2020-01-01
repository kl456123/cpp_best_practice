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
    Tensor(const std::vector<int>& tensor_shape, Tensor::DataType data_type=Tensor::DataType::FLOAT32,
            Backend::ForwardType type=Backend::ForwardType::CPU);
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

    static void ComputeStride(const std::vector<int>& shape, std::vector<int>& stride){
        stride.resize(shape.size(), 1);
        for(int i=shape.size()-2;i>=0;i--){
            stride[i] = stride[i+1]* shape[i+1];
        }
    }




    static int ComputeBufferSize(const int size, Tensor::DataType data_type){
        int buffer_size = 1;
        switch(data_type){
            case Tensor::DataType::FLOAT32:
                buffer_size = sizeof(float) * size;
                break;
            case Tensor::DataType::INT32:
                buffer_size = sizeof(int32_t) * size;
                break;
            case Tensor::DataType::DOUBLE:
                buffer_size = sizeof(double) * size;
                break;
            case Tensor::DataType::INT8:
                buffer_size = sizeof(int8_t) * size;
                break;
        }
        return buffer_size;
    }

    void CopyToDevice(Backend::ForwardType type_name){
        Backend* backend = ExtractBackend(type_name);
        backend->CopyFromHostToDevice(this);
        mDeviceType = type_name;
    }

    void CopyFromTensor(Tensor* other);

    void CopyToHost(){
        Backend* backend = ExtractBackend(mDeviceType);
        if(mHost==nullptr){
            Backend* cpu_backend = ExtractBackend(Backend::ForwardType::CPU);
            cpu_backend->Alloc(this);
        }
        backend->CopyFromDeviceToHost(this);
    }
    int Offset(int i, int j, int k, int l){
        std::vector<int> offset({i,j,k,l});
        return Offset(offset);
    }
    int Offset(int* offset){
        std::vector<int> offset_tmp(offset, offset+4);
        return Offset(offset_tmp);
    }

    int Offset(const std::vector<int>& offset){
        int index=0;
        for(int i=0;i<offset.size();i++){
            index+=offset[i]*mStride[i];
        }
        return index;
    }

    template<typename T>
        void Print(int size=-1);

    template<typename T>
        void Dump(const std::string& file_name);

    const std::vector<int>& stride(){
        return mStride;
    }

    void stride(int* stride){
        for(int i=0;i<mStride.size();i++){
            stride[i] = mStride[i];
        }
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
    template<typename T>
        inline T* device(){
            return (T*)mDevice;
        }

    DataType type(){
        return mDataType;
    }

    Backend::ForwardType device_type(){
        return mDeviceType;
    }


    private:
    void* mHost;
    void* mDevice;
    std::vector<int> mShape;
    std::vector<int> mStride;
    int mSize;
    size_t mBufferSize;
    bool mOwnMemory;
    // data type
    DataType mDataType;
    // device type
    Backend::ForwardType mDeviceType;

    // remove all assignment operator
    Tensor(const Tensor& tensor)  = delete;
    Tensor(const Tensor&& tensor) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor& operator=(const Tensor&&) = delete;

    void Init(const std::vector<int>& tensor_shape, DataType data_type);

};

#endif
