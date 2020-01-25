#ifndef CORE_OP_KERNEL_H_
#define CORE_OP_KERNEL_H_
#include <memory>
#include <unordered_map>
#include "core/device.h"
#include "node_def.pb.h"

/*
 * put Licensed here
 * */

class OpKernelContext;
class OpKernelConstruction;

class OpKernel{
    public:
        explicit OpKernel(OpKernelConstruction* context);
        virtual ~OpKernel();


        virtual void Compute(OpKernelContext* context)=0;


        //Accessor.
        const NodeDef& def()const{return def_;}
        const string& name()const;
        const string& requested_device()const;

        int num_inputs(){return input_types_.size();}
        int num_outputs(){return output_types_.size();}

    private:
        const std::unique_ptr<const NodeDef> def_;

        // memory type and data type for inputs and outputs
        const MemoryTypeVector input_memory_types_;
        const DataTypeVector input_types_;
        const MemoryTypeVector input_memory_types_;
        const DataTypeVector input_types_;

        std::ordered_map<std::string, int> input_name_map_;
        std::ordered_map<std::string, int> output_name_map_;
};


// allocate some memory when create op kernel
class OpKernelConstruction{
    public:


    private:
        // device and memory management
        const DeviceType device_type_;
        Device* const  device_;
        Allocator* allocator_;

        const NodeDef* def_;
        const OpDef* op_def_;

        friend class OpKernel;
};

// allocate some memory when run op kernel
class OpKernelContext{
    public:
        // used to initialize an OpKernelContext
        struct Params{
            OpKernel* op_kernel=nullptr;

            DeviceBase* device=nullptr;
            DeviceContext* op_device_context=nullptr;

            // session
            SessionState* session_state = nullptr;
            std::string session_handle;
            const SessionMetadata* session_metadata=nullptr;

            // inputs
            const std::vector<Tensor*>* inputs=nullptr;
            const std::vector<AllocatorAttributes>* input_alloc_attrs=nullptr;
        }

        // construction using params
        explicit OpKernelContext(Params* params);
        ~OpKernelContext();

        // accessor
        Env* env()const {return params_->device->env();}
        DeviceBase* device()const {return params_->device;}
        const OpKernel& op_kernel()const{return *params_->op_kernel;}
        DeviceContext* op_device_context(){
            DeviceContext* ret = params_->op_device_context;
            if(ret==nullptr){
                // default is gpu info
                auto* dev_info = device()->gpu_device_info();
                if(device_info)return dev_info->default_context;
            }
            return ret;
        }
        template<typename T>
            T* op_device_context();
        // input attrs and output attrs
        AllocatorAttributes input_alloc_attr(int index)const{
            if(params_->input_alloc_attrs==nullptr){
                return AllocatorAttributes();
            }else{
                return (*params_->input_alloc_attrs)[index];
            }
        }
        AllocatorAttributes output_alloc_attr(int index)const{
            return params_->output_attr_array[index];
        }
        // session accessor
        SessionState* session_state()const {return params_->session_state;}
        std::string session_handle()const{return params_->session_handle;}
        const SessionMetadata* session_metadata()const{return params_->session_metadata;}

        // status
        void SetStatus(const Status& status);
        const Status& status()const {return status_;}

        // inputs and outputs
        // memory type and data type
        int num_inputs()const {return params_->inputs->size();}
        DataType input_dtype(int index)const;
        Status input_dtype(std::string name, DataType* dtype)const;
        MemoryType input_memory_type(int index)const;

        int num_outputs()const {return outputs_.size();}
        DataType expected_output_type(int index)const;
        MemoryType output_memory_type(int index)const;

        // input
        Status input(int index, const Tensor** tensor);
        Status input(std::string name, const Tensor** tensor);
        Status mutable_input(int index, Tensor* tensor);
        Status mutable_input(std::string name, Tensor* tensor);

        // several allocate output methods
        Status allocate_output(int index, TensorShape& shape, Tensor** tensor);
        Status allocate_output(std::string name, TensorShape& shape, Tensor** tensor);

        Status allocate_temp(DataType type, const TensorShape& shape, Tensor* out_temp);

        Status allocate_persistent(DataType type, const TensorShape& shape,
                PersistentTensor* out_persistent, Tensor** out_tensor);

        // output
        Status set_output(std::string name, const Tensor& tensor);
        Status mutable_output(std::string name, Tensor** tensor);
    private:
        Status status_;
        Params* params_;
        // internal method to add a tensor's buffer to the list of buffers
        // referenced during the execution of the Op
        Status allocate_tensor(DataType type, const TensorShape& shape, Tensor* out_tensor);

        DISALLOW_COPY_AND_ASSIGN(OpKernelContext);

};

template<typename T>
T* OpKernelContext::op_device_context(){
    // assert T is subclass
    static_assert(std::is_base_of<DeviceContext, T>::value,
            "T is not a subclass of DeviceContext");
    return static_cast<T*>(op_device_context());
}

// Register OpKernel
//


#endif
