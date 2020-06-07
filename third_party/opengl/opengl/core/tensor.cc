#include <random>
#include <cmath>
#include "opengl/core/tensor.h"


namespace opengl{
    Tensor::~Tensor(){
        if(mem_type()==HOST_MEMORY){
            // host memory
            CHECK_NOTNULL(host_);
            CHECK_EQ(dtype(), DT_FLOAT);
            delete reinterpret_cast<float*>(host_);
        }else{
            // device memory
            delete reinterpret_cast<Texture*>(device_);
        }
    }

    template<>
        Tensor::Tensor(DataType dtype, IntList shape, float* data, DataFormat dformat)
        :shape_(shape),dtype_(dtype),mem_type_(HOST_MEMORY), dformat_(dformat){
            // AmendShape();
            CheckShapeAndDFormat();

            size_ = shape_.num_elements()*sizeof(float);
            host_ = data;

            initialized_=true;
        }

    Tensor::Tensor(const dlxnet::TensorProto& tensor_proto){
        // get tensor shape
        for(auto& dim:tensor_proto.dims()){
            shape_.add_dim(dim);
        }
        // AmendShape();
        // get tensor data
        switch(tensor_proto.data_type()){
            case dlxnet::TensorProto::FLOAT32:
                {
                    float* target_data = new float[num_elements()];
                    host_ = static_cast<void*>(target_data);
                    dtype_ = DT_FLOAT;
                    size_ = sizeof(float)*num_elements();
                    for(int i=0;i<num_elements();++i){
                        target_data[i] = tensor_proto.float_data(i);
                    }
                    break;
                }

            case dlxnet::TensorProto::INT32:
                {
                    int* target_data = new int[num_elements()];
                    host_ = static_cast<void*>(target_data);
                    size_ = sizeof(int)*num_elements();
                    dtype_ = DT_INT;
                    for(int i=0;i<num_elements();++i){
                        target_data[i] = tensor_proto.int32_data(i);
                    }
                    break;
                }

            default:
                LOG(FATAL)<<"unsupported const type: "<<tensor_proto.data_type();
        }
        initialized_=true;
        mem_type_ = HOST_MEMORY;
        dformat_=tensor_proto.data_format();
            CheckShapeAndDFormat();
    }

    // here we just allocate memory in host memory in hard code
    // TODO(breakpoint) add customs allocator input to allow
    // allocate device memory
    /*static*/ Tensor* Tensor::Empty(DataType dtype, IntList shape,
            DataFormat dformat){
        Tensor* tensor = new Tensor(dtype, shape, (float*)nullptr, dformat);
        const int num_elements = tensor->num_elements();
        float* image_data = new float[num_elements];
        tensor->set_host(image_data);
        return tensor;
    }

    /*static*/ Tensor* Tensor::Random(DataType dtype, IntList shape,
            DataFormat dformat){
        Tensor* tensor = Tensor::Empty(dtype, shape, dformat);
        float* data = tensor->host<float>();
        const int num_elements = tensor->num_elements();
        for(int i=0;i<num_elements;++i){
            data[i] = 1.0*random()/RAND_MAX;
        }
        return tensor;
    }

    /*static*/ Tensor* Tensor::Zeros(DataType dtype, IntList shape,
            DataFormat dformat){
        Tensor* tensor = Tensor::Empty(dtype, shape, dformat);
        memset(tensor->host(), 0, sizeof(float)*tensor->num_elements());
        return tensor;
    }

    /*static*/ Tensor* Tensor::Ones(DataType dtype, IntList shape,
            DataFormat dformat){
        Tensor* tensor = Tensor::Empty(dtype, shape, dformat);
        for(int i=0;i<tensor->num_elements();++i){
            tensor->host<float>()[i] = 1.0;
        }
        return tensor;
    }

    std::string Tensor::DebugString()const{
        CHECK(is_host());
        std::stringstream ss;
        ss<<"\n";
        auto&output_shape = shape();
        return ss.str();
    }

    std::string Tensor::ShortDebugString()const{
        CHECK(is_host());
        std::stringstream ss;
        ss<<"\n";
        const int num = num_elements();
        for(int i=0;i<num;++i){
            ss<< host<float>()[i] <<", ";
            if(num-i>10&&i>10){
                ss<<"..., ";
                i = num-10;
            }
        }
        return ss.str();
    }

    void Tensor::AsProto(dlxnet::TensorProto* proto)const{
        CHECK(is_host())<<"AsProto Only used in CPU Tensoor";
        // set shape
        for(auto dim: shape()){
            proto->add_dims(dim);
        }

        // set type
        proto->set_data_type(dlxnet::TensorProto::FLOAT32);
        proto->set_target_data_format(dlxnet::TensorProto::ANY4);
        proto->set_data_format(dformat());

        // set value
        const int num_elements_value = num_elements();
        proto->clear_float_data();
        const float* host_data = host<float>();
        for(int j=0;j<num_elements_value;++j){
            proto->add_float_data(host_data[j]);
        }
}
}//namespace opengl
