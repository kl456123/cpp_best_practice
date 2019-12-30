#ifndef CORE_TENSOR_H_
#define CORE_TENSOR_H_
#include <vector>
#include <memory>
#include <CL/cl.hpp>


class TensorShape;
class TensorBuffer;
class Backend;


class Tensor{
    struct TensorInfo{
        int dims;
    };


    public:
    class Filler{
        public:
            template<typename T>
                static void FillTensorByVal(Tensor* tensor,  float value){
                    auto data_type = tensor->mDataType;
                    for(int i=0;i<tensor->mSize;i++){
                        tensor->host<T>()[i] = value;
                    }
                }

            template<typename T>
                static void FillTensorRandomly(Tensor* tensor){
                    T MAX_VALUE=1000.0;
                    for(int i=0;i<tensor->mSize;i++){
                        tensor->host<T>()[i] = (rand()%int(MAX_VALUE)+1)/MAX_VALUE;
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

    template<typename T>
        static Tensor* Zeros(const std::vector<int>& tensor_shape,Tensor::DataType data_type);

    template<typename T>
        static Tensor* Ones(const std::vector<int>& tensor_shape,Tensor::DataType data_type);

    template<typename T>
        static Tensor* Random(const std::vector<int>& tensor_shape,Tensor::DataType data_type);

    static int ComputeSize(const std::vector<int>& shape){
        int size=1;
        for(int i=0;i<shape.size();i++){
            size*=shape[i];
        }
        return size;
    }


    inline std::vector<int> shape(){
        return mShape;
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

    inline size_t buffer_size(){
        return mBufferSize;
    }

    void set_host(void* ptr){
        mHost = ptr;
    }

    void set_device(void* ptr){
        mDevice = ptr;
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
